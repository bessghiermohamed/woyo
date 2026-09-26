"""Error taxonomy for woyo.

Errors are *data*, not control flow: tools never raise to the agent loop,
they return ToolResults carrying an ErrorKind so the model can reason about
what went wrong and choose a recovery strategy.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorKind(StrEnum):
    """Typed failure classification the executor reasons over."""

    TRANSIENT = "transient"            # timeout / rate limit — retry may help
    TOOL_FAILURE = "tool_failure"      # tool ran and failed — try differently
    INVALID_INPUT = "invalid_input"    # bad arguments — self-correct args
    MISSING_INFO = "missing_info"      # needed information absent — ask user / research
    NEEDS_APPROVAL = "needs_approval"  # user declined or must decide first
    IMPOSSIBLE = "impossible"          # cannot be done with available tools
    CONFIG = "config"                  # misconfiguration — fix setup


class ToolError(Exception):
    """Raised inside tools; converted to a typed ToolResult by the registry."""

    def __init__(self, kind: ErrorKind, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class AgentError(Exception):
    """Fatal error in the agent core (not a tool failure)."""


class BudgetExceeded(AgentError):
    """A run budget was exhausted; carries the reason for honest reporting."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
