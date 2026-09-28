"""Anthropic provider (optional dependency).

Converts woyo's canonical (OpenAI-shaped) messages to Anthropic's protocol:
- system messages become the top-level `system` parameter
- consecutive tool-result messages are merged into user turns with
  tool_result content blocks
- assistant tool_calls become tool_use content blocks
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from woyo.errors import AgentError
from woyo.models.base import Message, ModelResponse, ToolCall, ToolSpec, Usage

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover - optional extra
    _anthropic = None


def _supports_temperature() -> bool:
    """Cache whether this SDK generation accepts `temperature` (1.x dropped it)."""
    global _TEMP_SUPPORTED
    if _TEMP_SUPPORTED is None:
        import inspect

        try:
            params = inspect.signature(
                _anthropic.AsyncAnthropic().messages.create
            ).parameters
            _TEMP_SUPPORTED = "temperature" in params
        except Exception:  # noqa: BLE001 — be conservative
            _TEMP_SUPPORTED = False
    return _TEMP_SUPPORTED


_TEMP_SUPPORTED: bool | None = None


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: Any | None = None,
        timeout: float = 120.0,
    ):
        if _anthropic is None:
            raise AgentError(
                "anthropic package not installed. Run: pip install 'woyo[anthropic]'"
            )
        if not api_key:
            raise AgentError("No ANTHROPIC_API_KEY configured.")
        self._client = _anthropic.AsyncAnthropic(
            base_url=base_url, api_key=api_key, timeout=timeout, http_client=http_client
        )

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = 0.2,
        max_tokens: int | None = 4096,
    ) -> ModelResponse:
        system_text, converted = _convert_messages(messages)
        payload_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": converted,
                    "max_tokens": max_tokens or 4096,
                }
                if system_text:
                    kwargs["system"] = system_text
                # newer Anthropic SDKs removed `temperature` (thinking models);
                # pass it only when supported
                if temperature is not None and _supports_temperature():
                    kwargs["temperature"] = temperature
                if payload_tools:
                    kwargs["tools"] = payload_tools
                resp = await self._client.messages.create(**kwargs)
                return _from_anthropic_response(resp, model)
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                if status is None or status in (408, 409, 429, 500, 502, 503, 504):
                    last_error = exc
                else:
                    raise AgentError(f"Anthropic rejected the request: {exc}") from exc
            if attempt < 2:
                await asyncio.sleep(0.8 * (2**attempt))
        raise AgentError(f"Anthropic failed after retries: {last_error}")


def _parse_data_uri(uri: str) -> tuple[str | None, str]:
    """'data:image/jpeg;base64,AAAA' -> ('image/jpeg', 'AAAA')."""
    if not uri.startswith("data:"):
        return None, ""
    try:
        meta, _, payload = uri[5:].partition(",")
        mime = meta.split(";", 1)[0] or "image/jpeg"
        return mime, payload
    except Exception:  # noqa: BLE001 — malformed URIs are dropped, not fatal
        return None, ""


def _convert_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split system text; merge consecutive tool results into user turns."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
        elif m.role == "user":
            blocks: list[dict[str, Any]] = [
                {"type": "text", "text": m.content or ""}
            ]
            for uri in m.images or []:
                mime, b64 = _parse_data_uri(uri)
                if mime and b64:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        }
                    )
            converted.append({"role": "user", "content": blocks})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": args}
                )
            converted.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": [{"type": "text", "text": m.content or ""}],
            }
            if converted and converted[-1]["role"] == "user" and isinstance(
                converted[-1]["content"], list
            ) and converted[-1]["content"][0].get("type") == "tool_result":
                converted[-1]["content"].append(block)  # merge into same turn
            else:
                converted.append({"role": "user", "content": [block]})

    if not converted:
        converted.append({"role": "user", "content": [{"type": "text", "text": "(empty)"}]})
    return "\n\n".join(system_parts), converted


def _from_anthropic_response(resp: Any, model: str) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input or {}),
                )
            )
    usage = Usage(
        input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
    )
    finish = "tool_calls" if resp.stop_reason == "tool_use" else "stop"
    return ModelResponse(
        content="\n".join(text_parts) or None,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish,
        model=model,
    )
