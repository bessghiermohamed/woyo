"""End-to-end agent loop tests against the scripted MockProvider."""

from __future__ import annotations

from tests.conftest import (
    PLAN_JSON,
    FailingTool,
    build_agent,
)
from woyo.models.mock import text_response, tool_response
from woyo.tools.builtin.core_tools import FinishTool


async def test_happy_path_plan_tools_finish():
    agent, mock, bus = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "hello"})]),
            tool_response(
                [
                    (
                        "finish",
                        {
                            "summary": "The answer is hello-echoed.",
                            "verified": True,
                            "sources": [],
                            "open_questions": [],
                        },
                    )
                ]
            ),
        ]
    )
    result = await agent.run("say hello via echo then finish")

    assert result.outcome == "completed"
    assert result.final_answer == "The answer is hello-echoed."
    assert result.verified is True
    assert result.steps == 2  # two executor calls
    assert result.usage.tool_calls == 2  # echo + finish
    kinds = [e.kind for e in bus.events]
    assert "run_started" in kinds
    assert "plan_created" in kinds
    assert "run_finished" in kinds
    # executor received tool specs
    assert "finish" in mock.calls[1]["tool_names"]
    # observation made it into the conversation
    tool_msgs = [m for m in mock.calls[2]["messages"] if m.role == "tool"]
    assert any("echo: hello" in (m.content or "") for m in tool_msgs)


async def test_text_only_nudge_then_accept():
    agent, _, _ = build_agent(
        [
            text_response(PLAN_JSON),
            text_response("I will just answer directly."),
            text_response("The final answer is 42."),
        ]
    )
    result = await agent.run("what is the answer?")
    assert result.outcome == "completed"
    assert result.outcome_detail == "text-only final answer"
    assert result.final_answer == "The final answer is 42."
    assert result.verified is False


async def test_planner_garbage_falls_back():
    agent, _, bus = build_agent(
        [
            text_response("I cannot produce JSON, sorry!"),
            text_response("still not json"),
            tool_response(
                [("finish", {"summary": "Did it anyway, fallback plan used."})]
            ),
        ]
    )
    result = await agent.run("do the thing")
    assert result.outcome == "completed"
    assert result.plan is not None and len(result.plan.steps) == 1  # fallback
    kinds = [e.kind for e in bus.events]
    assert "plan_fallback" in kinds


async def test_unknown_tool_observation_then_finish():
    agent, mock, _ = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("does_not_exist", {})]),
            tool_response([("finish", {"summary": "Recovered from bad tool name."})]),
        ]
    )
    result = await agent.run("try a weird tool")
    assert result.outcome == "completed"
    tool_msgs = [m for m in mock.calls[2]["messages"] if m.role == "tool"]
    assert any("Unknown tool" in (m.content or "") for m in tool_msgs)


async def test_invalid_arguments_observation():
    agent, mock, _ = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("echo", {"wrong_field": 1})]),  # schema violation
            tool_response([("finish", {"summary": "Fixed my arguments afterwards."})]),
        ]
    )
    result = await agent.run("call echo badly")
    assert result.outcome == "completed"
    tool_msgs = [m for m in mock.calls[2]["messages"] if m.role == "tool"]
    assert any("Invalid arguments" in (m.content or "") for m in tool_msgs)


async def test_consecutive_failures_stop_run():
    agent, _, bus = build_agent(
        [text_response(PLAN_JSON)]
        + [
            tool_response([("failing_tool", {"note": f"attempt {n}"})])
            for n in range(4)
        ],
        tools=[FailingTool(), FinishTool()],
    )
    result = await agent.run("keep failing")
    assert result.outcome == "budget_exhausted"
    assert "consecutive tool failures" in result.outcome_detail
    assert result.verified is False
    assert "stopped before completing" in result.final_answer
    kinds = [e.kind for e in bus.events]
    assert "limit_hit" in kinds


async def test_executor_prompt_contains_security_contract():
    agent, mock, _ = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("finish", {"summary": "done immediately"})]),
        ]
    )
    await agent.run("check the system prompt")
    system = mock.calls[1]["messages"][0]
    assert system.role == "system"
    assert "<untrusted>" in system.content
    assert "NOT instructions" in system.content
    assert "never reveal secrets" in system.content.lower()
