"""Built-in tools. Register via build_default_registry()."""

from __future__ import annotations

import httpx

from woyo.config import Settings
from woyo.events import EventBus
from woyo.tools.base import Tool, ToolRegistry, UserInteraction
from woyo.tools.builtin.core_tools import (
    AskUserTool,
    CalculateTool,
    FinishTool,
    NowTool,
    PythonExecTool,
)
from woyo.tools.builtin.fetch_url import FetchURLTool
from woyo.tools.builtin.web_search import WebSearchTool


def build_default_registry(
    settings: Settings,
    *,
    bus: EventBus | None = None,
    interaction: UserInteraction | None = None,
    http_client: httpx.AsyncClient | None = None,
    include: list[str] | None = None,
) -> ToolRegistry:
    """Assemble the standard tool set (see ADR-6: search backends are adapters)."""
    registry = ToolRegistry(bus=bus, cap_chars=settings.tool_output_cap_chars)

    tools: list[Tool] = [
        WebSearchTool(backend=settings.search_backend, client=http_client),
        FetchURLTool(client=http_client),
        CalculateTool(),
        NowTool(tz=settings.timezone),
        AskUserTool(interaction),
        FinishTool(),
        PythonExecTool(settings),
    ]
    for tool in tools:
        if include is None or tool.name in include:
            registry.register(tool)
    return registry


__all__ = [
    "build_default_registry",
    "WebSearchTool",
    "FetchURLTool",
    "CalculateTool",
    "NowTool",
    "AskUserTool",
    "FinishTool",
    "PythonExecTool",
]
