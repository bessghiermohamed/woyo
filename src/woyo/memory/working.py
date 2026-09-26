"""Working memory: context-window management.

v0.1 compaction strategy (protocol-safe): when the estimated token count
exceeds the soft limit, the oldest bulky tool outputs are replaced with a
short placeholder. Message *structure* is never touched (every assistant
tool_call keeps its matching tool reply), so provider protocols stay valid.
Phase 2 replaces placeholdering with LLM summarization.
"""

from __future__ import annotations

from woyo.models.base import Message, estimate_tokens


def compact_context(messages: list[Message], soft_limit_tokens: int) -> int:
    """Compact in place; returns the number of tool outputs collapsed."""
    def total_est() -> int:
        return sum(estimate_tokens(m.content) for m in messages)

    if total_est() <= soft_limit_tokens:
        return 0

    collapsed = 0
    # oldest first; never touch the first two messages (system + task)
    for i in range(2, len(messages)):
        if total_est() <= soft_limit_tokens:
            break
        m = messages[i]
        if m.role == "tool" and m.content and len(m.content) > 400:
            m.content = (
                f"[compacted: earlier output of tool '{m.name}' was removed to "
                "save context. Re-fetch only if essential.]"
            )
            collapsed += 1
    return collapsed
