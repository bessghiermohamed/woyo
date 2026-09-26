"""OpenAI-compatible provider.

Covers OpenAI, Groq, OpenRouter, DeepSeek, Mistral, Ollama, vLLM, LM Studio,
and any custom OpenAI-shaped endpoint. Retries transient errors with backoff;
never retries auth/argument errors.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from woyo.errors import AgentError
from woyo.models.base import Message, ModelResponse, ToolCall, ToolSpec, Usage

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    name = "openai-compat"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: Any | None = None,  # inject httpx.AsyncClient(transport=...) in tests
        timeout: float = 120.0,
    ):
        if not api_key:
            raise AgentError(
                "No API key for the OpenAI-compatible provider. Set WOYO_API_KEY or the "
                "provider-native variable (see .env.example)."
            )
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )

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

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": payload_messages,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if payload_tools:
                    kwargs["tools"] = payload_tools
                resp = await self._client.chat.completions.create(**kwargs)
                return _from_openai_response(resp, model)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code in _TRANSIENT_STATUS:
                    last_error = exc
                else:
                    raise AgentError(f"Provider rejected the request: {exc}") from exc
            if attempt < 2:
                await asyncio.sleep(0.8 * (2**attempt))
        raise AgentError(f"Provider failed after retries: {last_error}")


def _to_openai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content or "",
        }
    payload: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    return payload


def _from_openai_response(resp: Any, model: str) -> ModelResponse:
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments or "{}",
        )
        for tc in (msg.tool_calls or [])
    ]
    usage = Usage()
    if resp.usage:
        usage.input_tokens = resp.usage.prompt_tokens or 0
        usage.output_tokens = resp.usage.completion_tokens or 0
    return ModelResponse(
        content=msg.content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.finish_reason,
        model=model,
    )


def safe_json_arguments(raw: str) -> str:
    """Normalize model-emitted tool arguments to a valid JSON object string."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "{}"
    if not isinstance(parsed, dict):
        return "{}"
    return json.dumps(parsed)
