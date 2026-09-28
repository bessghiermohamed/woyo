"""Memory integration: tools registered, recall injected, repeat-question test."""

from __future__ import annotations

import pytest

from tests.conftest import PLAN_JSON, build_agent, make_settings

pytestmark = pytest.mark.asyncio


async def _make_memory(tmp_path):
    from woyo.memory.longterm import MemoryStore

    store = MemoryStore(tmp_path / "mem.sqlite3", default_ttl_days=0)
    return store


class TestMemoryTools:
    async def test_registry_includes_memory_tools(self, tmp_path):
        from woyo.tools.builtin import build_default_registry

        settings = make_settings(memory_enabled=True)
        memory = await _make_memory(tmp_path)
        registry = build_default_registry(settings, memory=memory)
        assert "memory_save" in registry.names()
        assert "memory_search" in registry.names()
        memory.close()

    async def test_memory_save_then_search(self, tmp_path):
        from woyo.tools.builtin.memory_tools import build_memory_tools

        memory = await _make_memory(tmp_path)
        save_tool, search_tool = build_memory_tools(memory)

        from woyo.tools.builtin.memory_tools import MemorySaveArgs, MemorySearchArgs

        result = await save_tool.run(
            MemorySaveArgs(content="User's favorite editor is helix", kind="preference")
        )
        assert result.ok
        assert "saved to memory" in result.content

        result = await search_tool.run(MemorySearchArgs(query="favorite editor"))
        assert result.ok
        assert "helix" in result.content
        memory.close()

    async def test_memory_save_rejects_empty(self, tmp_path):
        from woyo.tools.builtin.memory_tools import MemorySave, MemorySaveArgs

        memory = await _make_memory(tmp_path)
        tool = MemorySave(memory)
        result = await tool.run(MemorySaveArgs(content="  ", kind="note"))
        assert not result.ok
        memory.close()


class TestRecallInjection:
    async def test_memory_recall_injects_into_prompt(self, tmp_path):
        """A stored fact must appear in the next run's system prompt."""
        from woyo.models.mock import text_response, tool_response

        memory = await _make_memory(tmp_path)
        await memory.remember(
            "The user's server is named tarzubal and runs Debian 12", kind="fact"
        )

        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([
                ("finish", {"summary": "tarzubal runs debian", "verified": True,
                            "sources": [], "open_questions": []}),
            ]),
        ])
        agent.memory = memory
        await agent.run("what OS does my server run?")
        executor_call = mock.calls[1]  # calls[0] is the planner
        system_prompt = executor_call["messages"][0].content
        assert "RELEVANT MEMORY" in system_prompt
        assert "tarzubal" in system_prompt
        assert "NOT verified sources" in system_prompt  # untrusted-hints framing
        memory.close()

    async def test_no_memory_section_when_store_empty(self, tmp_path):
        from woyo.models.mock import text_response, tool_response

        memory = await _make_memory(tmp_path)
        agent, mock, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([
                ("finish", {"summary": "x", "verified": True, "sources": [],
                            "open_questions": []}),
            ]),
        ])
        agent.memory = memory
        await agent.run("an unrelated brand new question")
        executor_call = mock.calls[1]  # calls[0] is the planner
        system_prompt = executor_call["messages"][0].content
        assert "RELEVANT MEMORY" not in system_prompt
        memory.close()


class TestRepeatQuestion:
    async def test_followup_task_gets_the_fact_without_new_tools(self, tmp_path):
        """Phase 3 exit criterion: memory measurably helps a follow-up.

        Run 1 stores a fact via the memory_save tool (scripted). Run 2 is a
        fresh agent on a related question: the fact is already in its system
        prompt, so it can finish with zero tool calls.
        """
        from woyo.models.mock import text_response, tool_response

        # --- run 1: the agent learns and saves a fact --------------------
        memory = await _make_memory(tmp_path)
        agent1, _, _ = build_agent([
            text_response(PLAN_JSON),
            tool_response([
                ("memory_save", {"content": "deploy.yml pins woyo version 0.3.2",
                                 "kind": "fact"}),
            ]),
            tool_response([
                ("finish", {"summary": "saved the deploy version fact",
                            "verified": True, "sources": [], "open_questions": []}),
            ]),
        ])
        agent1.memory = memory
        from woyo.tools.builtin import build_default_registry

        # give the registry the memory tools for run 1
        settings = make_settings(memory_enabled=True)
        registry = build_default_registry(settings, memory=memory)
        agent1.registry = registry
        result1 = await agent1.run("remember which woyo version deploy.yml pins")
        assert result1.outcome == "completed"
        assert memory.stats()["total"] == 1

        # --- run 2: fresh agent (fresh everything but the store) ---------
        agent2, mock2, _ = build_agent([
            text_response(PLAN_JSON),
            text_response("deploy.yml pins woyo 0.3.2 (from memory)"),
        ])
        agent2.memory = memory
        agent2.settings.direct_text_replies = True
        result2 = await agent2.run("which woyo version does deploy.yml pin?")
        assert result2.outcome == "completed"
        assert "0.3.2" in result2.final_answer
        # measurable: the fact arrived via the prompt, not via any tool
        assert result2.usage.tool_calls == 0
        system_prompt = mock2.calls[1]["messages"][0].content  # executor, not planner
        assert "deploy.yml pins woyo version 0.3.2" in system_prompt
        memory.close()
