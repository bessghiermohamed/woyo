"""Checkpointing, resume, and control: the loop survives restarts (Phase 3).

The "restart" in these tests is a fresh Agent instance with a fresh
MockProvider — exactly what a resumed process looks like: only the
checkpoint dict crosses the boundary.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    PLAN_JSON,
    build_agent,
    default_test_tools,
    make_registry,
)
from woyo.agent import Agent
from woyo.agent.state import RunState
from woyo.models.mock import MockProvider, text_response, tool_response
from woyo.models.router import ModelRouter


def _finish_call(summary: str, verified: bool = True):
    assert len(summary) >= 10, "finish summaries must pass min_length=10"
    return tool_response([
        ("finish", {"summary": summary, "verified": verified, "sources": [],
                    "open_questions": []}),
    ])


class TestRunState:
    def test_roundtrip_preserves_everything(self):
        state = RunState(
            task="do a thing",
            system_prompt="sys",
            plan_text="plan",
            messages=[],
            steps=3,
            tool_calls=5,
            search_calls=2,
            consecutive_failures=1,
            text_only_strikes=0,
            observed_urls={"https://a.example", "https://b.example"},
            elapsed_s=12.5,
        )
        state.repeat_counter[("echo", "{}")] = 2
        raw = state.to_json()
        restored = RunState.from_json(raw)
        assert restored.task == "do a thing"
        assert restored.steps == 3
        assert restored.tool_calls == 5
        assert restored.search_calls == 2
        assert restored.observed_urls == {"https://a.example", "https://b.example"}
        assert restored.elapsed_s == 12.5
        assert restored.repeat_counter[("echo", "{}")] == 2

    def test_roundtrip_preserves_tool_call_messages(self):
        state = RunState(task="t", messages=[])
        from woyo.models.base import Message

        state.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="Task: t"),
            Message(role="assistant", content=None, tool_calls=[
                __import__("woyo.models.base", fromlist=["ToolCall"]).ToolCall(
                    id="call-1", name="echo", arguments='{"text": "hi"}'
                ),
            ]),
            Message(role="tool", tool_call_id="call-1", name="echo", content="echo: hi"),
        ]
        restored = RunState.from_dict(RunState.from_dict(state.to_dict()).to_dict())
        assert restored.messages[2].tool_calls[0].name == "echo"
        assert restored.messages[3].tool_call_id == "call-1"


class TestCheckpointing:
    async def test_checkpoint_fires_each_step(self):
        checkpoints: list[dict] = []
        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "one"})]),
            _finish_call("all done, verified"),
        ])
        await agent.run("test", checkpoint_cb=checkpoints.append)
        assert len(checkpoints) >= 2
        assert all("steps" in c and "messages" in c for c in checkpoints)
        assert checkpoints[-1]["steps"] == 2  # two executor steps consumed

    async def test_checkpoint_survives_checkpoint_cb_crash(self):
        def bad_cb(_):
            raise RuntimeError("disk on fire")

        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            _finish_call("done anyway, checkpoint crash survived"),
        ])
        result = await agent.run("test", checkpoint_cb=bad_cb)
        assert result.outcome == "completed"


class TestControl:
    async def test_cancel_between_steps(self):
        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "one"})]),
            tool_response([("echo", {"text": "two"})]),
            _finish_call("never reached"),
        ])
        calls = {"n": 0}

        def control():
            calls["n"] += 1
            return "cancel" if calls["n"] > 1 else None

        result = await agent.run("test", control_poll=control)
        assert result.outcome == "cancelled"
        assert "cancelled by the user" in result.final_answer

    async def test_pause_between_steps(self):
        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "one"})]),
            tool_response([("echo", {"text": "two"})]),
            _finish_call("never reached"),
        ])
        calls = {"n": 0}

        def control():
            calls["n"] += 1
            return "pause" if calls["n"] > 1 else None

        result = await agent.run("test", control_poll=control)
        assert result.outcome == "paused"
        assert "paused" in result.final_answer


class TestResume:
    async def test_task_survives_restart(self):
        """The Phase 3 exit criterion: run, kill mid-flight, resume, complete."""
        # --- process 1: plan + one echo step, then "crash" (cancel) -----
        checkpoints: list[dict] = []
        agent1, mock1, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "hello from process one"})]),
            tool_response([("echo", {"text": "never seen"})]),
            _finish_call("never reached"),
        ])
        state = {"polls": 0}

        def cancel_after_first_step():
            state["polls"] += 1
            return "cancel" if state["polls"] > 1 else None

        result1 = await agent1.run(
            "survive a restart", checkpoint_cb=checkpoints.append,
            control_poll=cancel_after_first_step,
        )
        assert result1.outcome == "cancelled"
        assert checkpoints, "expected at least one checkpoint before the cancel"

        # --- the "restart": brand-new agent, only the checkpoint remains --
        agent2, mock2, _ = build_agent([
            tool_response([("echo", {"text": "hello from process two"})]),
            _finish_call("resumed and finished"),
        ])
        result2 = await agent2.run(
            "survive a restart", resume_state=checkpoints[-1]
        )
        assert result2.outcome == "completed"
        assert result2.final_answer == "resumed and finished"
        # the first executor message of process 2 continues the same
        # conversation: system + task + prior assistant/tool messages present
        first_messages = mock2.calls[0]["messages"]
        roles = [m.role for m in first_messages]
        assert roles[0] == "system"
        assert "process one" in first_messages[-1].content  # prior observation kept
        # and it did NOT restart from the plan: no planner JSON was consumed
        # 1 step from p1 + (echo + finish) = 2 executor steps in p2
        assert result2.steps == 3

    async def test_resume_rejects_different_task(self):
        checkpoints: list[dict] = []
        agent, _, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "x"})]),
            tool_response([("echo", {"text": "y"})]),
            _finish_call("unreachable finish z"),
        ])
        await agent.run("original task", checkpoint_cb=checkpoints.append)
        other, _, _ = build_agent([_finish_call("nope not this one")])
        from woyo.errors import AgentError

        with pytest.raises(AgentError, match="different task"):
            await other.run("a different task", resume_state=checkpoints[-1])

    async def test_elapsed_budget_carries_over(self):
        """A resumed run must not get a fresh time budget."""
        checkpoints: list[dict] = []
        agent1, _, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "one"})]),
            tool_response([("echo", {"text": "two"})]),
            _finish_call("unreachable finish x"),
        ])

        polls = {"n": 0}

        def pause_after_first_step():
            polls["n"] += 1
            return "pause" if polls["n"] > 1 else None

        await agent1.run("budget test", checkpoint_cb=checkpoints.append,
                         control_poll=pause_after_first_step)
        # pretend the checkpoint says 999s were already spent
        cp = checkpoints[-1]
        cp["elapsed_s"] = 999.0
        agent2, _, _ = build_agent([_finish_call("should stop fast")])
        result = await agent2.run("budget test", resume_state=cp)
        assert result.outcome == "budget_exhausted"
        assert "time limit" in result.outcome_detail


class TestTaskRunner:
    async def test_runner_persists_lifecycle(self, tmp_path):
        from woyo.agent.runner import TaskRunner
        from woyo.store.tasks import TaskStore

        store = TaskStore(tmp_path / "tasks.sqlite3")
        task = store.create("runner test task")

        def factory(settings, bus):
            mock = MockProvider([
                text_response(PLAN_JSON),
                _finish_call("runner done, completed"),
            ])
            router = ModelRouter(settings, default_provider=mock, bus=bus)
            registry = make_registry(default_test_tools(), bus=bus)
            return Agent(settings, router, registry, bus=bus)

        runner = TaskRunner(build_agent([])[0].settings, store, agent_factory=factory)
        result = await runner.run_task(task.id)
        assert result.outcome == "completed"

        row = store.get(task.id)
        assert row.status == "completed"
        assert row.result["final_answer"] == "runner done, completed"
        assert row.attempts == 1
        kinds = [e["kind"] for e in store.events(task.id)]
        assert "run_started" in kinds
        assert "run_finished" in kinds
        store.close()

    async def test_runner_marks_crash_as_failed(self, tmp_path):
        from woyo.agent.runner import TaskRunner
        from woyo.store.tasks import TaskStore

        store = TaskStore(tmp_path / "tasks.sqlite3")
        task = store.create("boom")

        def factory(settings, bus):
            mock = MockProvider([
                text_response(PLAN_JSON),
                RuntimeError("provider exploded"),
            ])
            router = ModelRouter(settings, default_provider=mock, bus=bus)
            registry = make_registry(default_test_tools(), bus=bus)
            return Agent(settings, router, registry, bus=bus)

        settings = build_agent([])[0].settings
        runner = TaskRunner(settings, store, agent_factory=factory)
        with pytest.raises(RuntimeError):
            await runner.run_task(task.id)
        assert store.get(task.id).status == "failed"
        assert "provider exploded" in store.get(task.id).error
        store.close()

    async def test_runner_pause_via_control(self, tmp_path):
        from woyo.agent.runner import TaskRunner
        from woyo.store.tasks import TaskStore

        store = TaskStore(tmp_path / "tasks.sqlite3")
        task = store.create("pausable")

        def factory(settings, bus):
            agent, _, _ = build_agent([
                text_response(PLAN_JSON),
                tool_response([("echo", {"text": "one"})]),
                tool_response([("echo", {"text": "two"})]),
                _finish_call("unreachable finish x"),
            ])
            return agent

        polls = {"n": 0}

        # pause the task from "another process" after the first step
        class PausingStore(TaskStore):
            def poll_control(self, task_id):
                polls["n"] += 1
                return "pause" if polls["n"] > 1 else None

        pausing = PausingStore(store.path)
        settings = build_agent([])[0].settings
        runner = TaskRunner(settings, pausing, agent_factory=factory)
        result = await runner.run_task(task.id)
        assert result.outcome == "paused"
        assert pausing.get(task.id).status == "paused"
        assert pausing.get(task.id).checkpoint is not None
        pausing.close()
        store.close()
