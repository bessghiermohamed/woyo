"""GPT4Free (g4f) provider — experimental, opt-in, never a default.

Wraps the `g4f` library (github.com/xtekky/gpt4free), which routes requests to
free LLM endpoints collected from the web. It is OpenAI-SDK-shaped, so this
adapter reuses the openai_compat message/response conversion.

Why this is opt-in and clearly labelled experimental (see docs/DECISIONS.md,
ADR-9 and docs/SECURITY.md):
  * Reliability: providers appear and disappear without notice; no SLA.
  * Terms of service: some upstream endpoints do not sanction automated use.
  * Data safety: prompts travel to unvetted third parties — never send
    secrets, personal data, or anything confidential through g4f.
  * The library is heavyweight (browser automation deps) — hence an extra.

Install with:  pip install "woyo[g4f]"
Use with:      WOYO_PROVIDER=g4f WOYO_MODEL=<g4f model id>
"""

from __future__ import annotations

import asyncio

from woyo.errors import AgentError
from woyo.models.base import Message, ModelResponse, ToolSpec
from woyo.models.openai_compat import _from_openai_response, _to_openai_message


class G4fProvider:
    name = "g4f"

    def __init__(self) -> None:
        try:
            from g4f.client import Client as G4fClient
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise AgentError(
                "The g4f extra is not installed. Run: pip install 'woyo[g4f]'"
            ) from exc
        self._client = G4fClient()

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        payload_messages = [_to_openai_message(m) for m in messages]
        payload_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        kwargs: dict[str, object] = {"model": model, "messages": payload_messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if payload_tools:
            kwargs["tools"] = payload_tools

        def _call():
            # g4f's client is sync; it also retries/falls back between
            # providers internally, so we do not add our own retry loop.
            return self._client.chat.completions.create(**kwargs)

        try:
            resp = await asyncio.to_thread(_call)
        except Exception as exc:  # noqa: BLE001 - g4f raises many shapes
            raise AgentError(f"g4f request failed: {exc}") from exc
        return _from_openai_response(resp, model)
