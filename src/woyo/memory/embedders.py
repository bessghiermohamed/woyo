"""Embeddings for long-term memory recall (Phase 3, ADR-11).

Two implementations behind one protocol:

- HashEmbedder — deterministic, offline, zero-dependency. A hashing trick
  over word unigrams+bigrams projected to 512 dims, L2-normalized. Good
  lexical similarity: enough to recall "the Python question" for "python
  version?", which is what a personal assistant's memory needs. ALWAYS
  available, so memory works with mock/cohere/any provider.

- OpenAICompatEmbedder — real embeddings via the provider's
  /v1/embeddings endpoint (openai, gemini, ollama, deepseek, custom).
  Providers without an OpenAI-compatible embeddings route (cohere's
  compat layer, openrouter, ...) fall back to HashEmbedder automatically.

Selection: WOYO_MEMORY_EMBEDDER = auto | hash | openai.
`auto` prefers the API embedder when the provider is known to support it
and falls back to hash on any failure (sticky for the process lifetime).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: providers whose OpenAI-compat endpoint serves /embeddings
_API_EMBED_PROVIDERS = {"openai", "gemini", "ollama", "deepseek", "custom"}
_DEFAULT_EMBED_MODEL = {
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
    "ollama": "nomic-embed-text",
    "deepseek": "text-embedding-3-small",
    "custom": "text-embedding-3-small",
}


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec
    grams: list[str] = []
    grams += tokens  # unigrams
    grams += [f"{a}~{b}" for a, b in zip(tokens, tokens[1:], strict=False)]  # bigrams
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], "little") % _DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        weight = 1.0 / math.sqrt(len(gram))  # longer grams count a bit less
        vec[slot] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class HashEmbedder:
    """Deterministic offline embeddings — the default and the fallback."""

    name = "hash-512"
    dim = _DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]


class OpenAICompatEmbedder:
    """Embeddings via an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, *, base_url: str | None, api_key: str | None, model: str):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = 0  # learned from the first response
        self.name = f"api:{model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            payload = resp.json()
        vectors = [item["embedding"] for item in payload["data"]]
        self.dim = len(vectors[0]) if vectors else 0
        return vectors


def build_embedder(settings) -> Embedder:
    """Resolve the embedder from settings (auto | hash | openai)."""
    choice = (settings.memory_embedder or "auto").lower()
    provider = (settings.provider or "").lower()

    if choice in ("auto", "openai", "api") and provider in _API_EMBED_PROVIDERS:
        from woyo.config import resolve_api_key, resolve_base_url

        model = settings.memory_embed_model or _DEFAULT_EMBED_MODEL.get(
            provider, "text-embedding-3-small"
        )
        return OpenAICompatEmbedder(
            base_url=resolve_base_url(provider, settings),
            api_key=resolve_api_key(provider, settings),
            model=model,
        )
    # explicit openai choice on a provider we don't know serves embeddings:
    # still try, with the bare api key, unless the user pinned 'hash'.
    if choice in ("openai", "api") and provider not in _API_EMBED_PROVIDERS:
        if settings.memory_embed_model:
            from woyo.config import resolve_api_key, resolve_base_url

            return OpenAICompatEmbedder(
                base_url=resolve_base_url(provider, settings),
                api_key=resolve_api_key(provider, settings),
                model=settings.memory_embed_model,
            )
    return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for equal-length dense vectors (0 when lengths differ)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
