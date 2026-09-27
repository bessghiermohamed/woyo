"""woyo — a general-purpose AI agent you own.

Understands goals, plans, uses tools, verifies results, and reports back —
with human control built in.
"""

__version__ = "0.1.1"

from woyo.agent.loop import Agent
from woyo.agent.results import RunResult
from woyo.config import Settings
from woyo.events import Event, EventBus
from woyo.models.router import ModelRouter
from woyo.tools.base import Permission, Tool, ToolRegistry, ToolResult

__all__ = [
    "__version__",
    "Agent",
    "RunResult",
    "Settings",
    "Event",
    "EventBus",
    "ModelRouter",
    "Permission",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
