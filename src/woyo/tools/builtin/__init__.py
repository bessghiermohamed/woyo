"""Built-in tools. Register via build_default_registry()."""

from __future__ import annotations

import httpx

from woyo.config import Settings
from woyo.events import EventBus
from woyo.memory.longterm import MemoryStore
from woyo.tools.base import Tool, ToolRegistry, UserInteraction
from woyo.tools.builtin.agent_tool import SpawnAgentTool
from woyo.tools.builtin.core_tools import (
    AskUserTool,
    CalculateTool,
    FinishTool,
    NowTool,
    PythonExecTool,
)
from woyo.tools.builtin.crawl_site import CrawlSiteTool
from woyo.tools.builtin.documents import CreateDocumentTool
from woyo.tools.builtin.fetch_url import FetchURLTool
from woyo.tools.builtin.file_transfer import FileSender, SendFileTool
from woyo.tools.builtin.sandbox import (
    ListDirTool,
    ReadFileTool,
    ShellExecTool,
    WriteFileTool,
)
from woyo.tools.builtin.schedule_tools import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)
from woyo.tools.builtin.telegram_tools import (
    GetChatInfoTool,
    ListChatsTool,
    SendDocumentTool,
    SendMessageTool,
    TelegramTransport,
)
from woyo.tools.builtin.web_search import WebSearchTool
from woyo.tools.http_cache import build_cache_from_settings


def build_default_registry(
    settings: Settings,
    *,
    bus: EventBus | None = None,
    interaction: UserInteraction | None = None,
    http_client: httpx.AsyncClient | None = None,
    include: list[str] | None = None,
    memory: MemoryStore | None = None,
    router=None,  # ModelRouter — enables spawn_agent (sub-agents)
    file_sender: FileSender | None = None,  # chat transport — enables send_file
    telegram: TelegramTransport | None = None,  # enables the telegram tools
    job_store=None,  # JobStore — enables the scheduler tools
    chat_id: int | None = None,  # the chat the scheduler tools act for
) -> ToolRegistry:
    """Assemble the standard tool set (see ADR-6: search backends are adapters)."""
    registry = ToolRegistry(bus=bus, cap_chars=settings.tool_output_cap_chars)
    cache = build_cache_from_settings(settings)

    tools: list[Tool] = [
        WebSearchTool(
            backend=settings.search_backend,
            client=http_client,
            cache=cache,
            cache_search_ttl_s=settings.cache_search_ttl_s,
        ),
        FetchURLTool(client=http_client, cache=cache),
        CrawlSiteTool(client=http_client, cache=cache),
        CalculateTool(),
        NowTool(tz=settings.timezone),
        AskUserTool(interaction),
        FinishTool(),
        PythonExecTool(settings),
        ShellExecTool(settings),
        ReadFileTool(settings),
        WriteFileTool(settings),
        ListDirTool(settings),
        CreateDocumentTool(settings),
    ]
    if router is not None:
        # sub-agents share the caller's router: one budget, one cost account
        tools.append(SpawnAgentTool(settings, router, bus=bus, memory=memory))
    if file_sender is not None:
        # chat frontends only: deliver workspace files to the requesting chat
        tools.append(SendFileTool(settings, file_sender))
    if telegram is not None:
        # chat frontends with a Bot API transport: addressable sends
        tools += [
            SendMessageTool(telegram),
            ListChatsTool(telegram),
            GetChatInfoTool(telegram),
            SendDocumentTool(settings, telegram),
        ]
    if job_store is not None and chat_id is not None:
        # chat frontends with a scheduler: promises become rows
        tools += [
            ScheduleTaskTool(settings, job_store, chat_id),
            ListScheduledTasksTool(settings, job_store, chat_id),
            CancelScheduledTaskTool(job_store, chat_id),
        ]
    if memory is not None:
        from woyo.tools.builtin.memory_tools import build_memory_tools

        tools += build_memory_tools(memory, bus=bus)
    for tool in tools:
        if include is None or tool.name in include:
            registry.register(tool)
    return registry


__all__ = [
    "build_default_registry",
    "WebSearchTool",
    "FetchURLTool",
    "CrawlSiteTool",
    "CalculateTool",
    "NowTool",
    "AskUserTool",
    "FinishTool",
    "PythonExecTool",
    "ShellExecTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "CreateDocumentTool",
    "SpawnAgentTool",
    "SendFileTool",
    "SendMessageTool",
    "ListChatsTool",
    "GetChatInfoTool",
    "SendDocumentTool",
    "ScheduleTaskTool",
    "ListScheduledTasksTool",
    "CancelScheduledTaskTool",
]
