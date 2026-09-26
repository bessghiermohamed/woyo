"""Model router: role -> (provider, model), with usage & cost accounting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from woyo.config import Role, Settings
from woyo.events import EventBus
from woyo.models.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ToolSpec,
    Usage,
    estimate_tokens,
)
from woyo.models.costs import estimate_cost_usd


def split_model_ref(ref: str) -> tuple[str | None, str]:
    """"groq:llama-3.3-70b" -> ("groq", "llama-3.3-70b"); bare -> (None, ref)."""
    if ":" in ref and not ref.startswith(("http://", "https://")):
        provider, model = ref.split(":", 1)
        return provider, model
    return None, ref


class ModelRouter:
    """Resolves which provider+model serves each role; tracks usage and cost."""

    def __init__(
        self,
        settings: Settings,
        *,
        default_provider: ModelProvider | None = None,
        extra_providers: dict[str, ModelProvider] | None = None,
        bus: EventBus | None = None,
    ):
        self.settings = settings
        self.bus = bus
        self._providers: dict[str, ModelProvider] = dict(extra_providers or {})
        self._default = default_provider
        self._usage_total = Usage()
        self._usage_by_model: dict[str, Usage] = defaultdict(Usage)
        self._cost_usd = 0.0

    # ------------------------------------------------------------------
    def _provider_for(self, provider_name: str | None) -> ModelProvider:
        if provider_name is None:
            if self._default is not None:
                return self._default
            return self._build_default_provider()
        if provider_name in self._providers:
            return self._providers[provider_name]
        provider = self._build_named_provider(provider_name)
        self._providers[provider_name] = provider
        return provider

    def _build_default_provider(self) -> ModelProvider:
        provider = self._build_named_provider(self.settings.provider)
        if self._default is None:
            self._default = provider
        return provider

    def _build_named_provider(self, name: str) -> ModelProvider:
        # Local/deferred imports keep optional deps optional.
        if name == "mock":
            from woyo.models.mock import MockProvider

            return MockProvider(responses=[])
        if name == "anthropic":
            from woyo.config import resolve_api_key
            from woyo.models.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                base_url=self.settings.base_url if name == self.settings.provider else None,
                api_key=resolve_api_key("anthropic", self.settings),
            )
        from woyo.config import resolve_api_key, resolve_base_url
        from woyo.models.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=resolve_base_url(name, self.settings),
            api_key=resolve_api_key(name, self.settings),
        )

    # ------------------------------------------------------------------
    async def complete(
        self,
        role: Role,
        *,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        ref = self.settings.model_for_role(role)
        provider_name, model = split_model_ref(ref)
        provider = self._provider_for(provider_name)
        est_in = sum(estimate_tokens(m.content) for m in messages)
        if self.bus:
            self.bus.emit("llm_call", role=role, model=model, est_input_tokens=est_in)
        response = await provider.complete(
            messages=messages,
            tools=tools or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._usage_total.add(response.usage)
        self._usage_total.calls += 1
        self._usage_by_model[model].add(response.usage)
        self._usage_by_model[model].calls += 1
        self._cost_usd += estimate_cost_usd(model, response.usage)
        return response

    # ------------------------------------------------------------------
    @property
    def total_usage(self) -> Usage:
        return self._usage_total

    def usage_summary(self) -> dict[str, Any]:
        by_model = {
            model: {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "calls": u.calls,
            }
            for model, u in self._usage_by_model.items()
        }
        return {
            "total": {
                "input_tokens": self._usage_total.input_tokens,
                "output_tokens": self._usage_total.output_tokens,
                "calls": self._usage_total.calls,
            },
            "by_model": by_model,
            "cost_usd_est": round(self._cost_usd, 6),
        }

    @property
    def cost_usd_est(self) -> float:
        return self._cost_usd
