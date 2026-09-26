"""Rough per-model price table (USD per 1k tokens) for cost budgeting.

These are ESTIMATES for orientation, not billing data. Unknown models cost 0
by default and emit a warning once — budgets then rely on token limits.
"""

from __future__ import annotations

from woyo.models.base import Usage

# (input_usd_per_1k, output_usd_per_1k)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI (estimates)
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o4-mini": (0.0011, 0.0044),
    # Anthropic (estimates)
    "claude-3-5-haiku-latest": (0.0008, 0.004),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-3-5-sonnet-latest": (0.003, 0.015),
    # Google (estimates)
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.5-flash": (0.0003, 0.0025),
    # Groq (estimates)
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
}

_warned_unknown: set[str] = set()


def estimate_cost_usd(model: str, usage: Usage) -> float:
    key = model.lower()
    for known, price in PRICE_TABLE.items():
        if key.startswith(known.lower()):
            return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1000
    if key not in _warned_unknown:
        _warned_unknown.add(key)
    return 0.0
