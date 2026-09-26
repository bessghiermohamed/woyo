"""Budget & limit enforcement: steps, tool calls, loops, tokens."""

from __future__ import annotations

from tests.conftest import PLAN_JSON, build_agent, make_settings
from woyo.models.mock import text_response, tool_response


async def test_step_limit_produces_honest_partial():
    agent, _, bus = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("echo", {"text": "one"})]),
            tool_response([("echo", {"text": "two"})]),  # never reached
        ],
        settings=make_settings(max_steps=1),
    )
    result = await agent.run("do several steps")
    assert result.outcome == "budget_exhausted"
    assert "step limit" in result.outcome_detail
    assert result.verified is False
    assert "stopped before completing" in result.final_answer
    assert any(e.kind == "limit_hit" for e in bus.events)


async def test_tool_call_limit_mid_batch():
    agent, _, _ = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response(
                [("echo", {"text": "a"}), ("echo", {"text": "b"})]
            ),  # 2 calls, limit is 1
        ],
        settings=make_settings(max_tool_calls=1),
    )
    result = await agent.run("call tools")
    assert result.outcome == "budget_exhausted"
    assert "tool-call limit" in result.outcome_detail


async def test_loop_detection_on_identical_calls():
    agent, _, bus = build_agent(
        [text_response(PLAN_JSON)]
        + [tool_response([("echo", {"text": "same"})]) for _ in range(4)],
    )
    result = await agent.run("loop forever")
    assert result.outcome == "budget_exhausted"
    assert "loop detected" in result.outcome_detail
    # the 3rd identical call got a warning observation, not an execution
    kinds = [e.kind for e in bus.events]
    assert "notice" in kinds


async def test_time_limit():
    agent, _, _ = build_agent(
        [text_response(PLAN_JSON)]
        + [tool_response([("echo", {"text": "x"})]) for _ in range(6)],
        settings=make_settings(max_time_s=0.0),  # immediate exhaustion
    )
    result = await agent.run("beat the clock")
    assert result.outcome == "budget_exhausted"
    assert "time limit" in result.outcome_detail


async def test_budget_report_in_run_result():
    agent, _, _ = build_agent(
        [
            text_response(PLAN_JSON),
            tool_response([("finish", {"summary": "Quick and cheap."})]),
        ],
    )
    result = await agent.run("cheap task")
    assert result.usage.llm_calls == 2  # planner + executor
    assert result.usage.tool_calls == 1
    assert result.duration_s >= 0
    assert result.steps == 1
