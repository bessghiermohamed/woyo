"""Configuration via environment / .env (pydantic-settings).

Provider resolution is explicit and predictable:
  WOYO_PROVIDER  ->  preset that fixes base_url + which env var carries the key
Model references may be bare ("gpt-4o-mini", default provider) or
prefixed ("groq:llama-3.3-70b-versatile", that provider).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["planner", "executor", "summarizer"]

PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    # provider -> {base_url, key_env, key_default}
    # key_env may list several env vars (comma-separated) — first match wins.
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "key_default": None},
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_default": None,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY,GOOGLE_API_KEY",
        "key_default": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_default": None,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "key_default": None,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "key_default": None,
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "key_env": "XAI_API_KEY",
        "key_default": None,
    },
    "cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "key_env": "COHERE_API_KEY",
        "key_default": None,
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN,HUGGINGFACE_API_KEY",
        "key_default": None,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "key_default": "ollama",
    },
    "anthropic": {"base_url": None, "key_env": "ANTHROPIC_API_KEY", "key_default": None},
    "custom": {"base_url": "WOYO_BASE_URL", "key_env": "WOYO_API_KEY", "key_default": None},
    "g4f": {"base_url": None, "key_env": None, "key_default": "g4f"},
    "mock": {"base_url": None, "key_env": None, "key_default": "mock"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WOYO_", env_file=".env", extra="ignore")

    # --- model layer ---
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    planner_model: str | None = None
    executor_model: str | None = None
    summarizer_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    # --- budgets ---
    max_steps: int = 15
    max_tool_calls: int = 30
    max_time_s: float = 600.0
    max_tokens: int = 400_000
    max_cost_usd: float = 1.0

    # --- safety ---
    require_approval: bool = True
    enable_code_exec: bool = False
    enable_shell: bool = False  # shell_exec: approval AND this flag (double deny)

    # --- search ---
    search_backend: str | None = None  # tavily | brave | ddg (None = auto)
    max_search_calls: int = 12  # per task: web_search + crawl_site combined

    # --- research cache (Phase 2) ---
    cache_enabled: bool = True
    cache_dir: str = "~/.woyo"
    cache_ttl_s: int = 86_400       # fetched pages: 24h
    cache_search_ttl_s: int = 3_600  # search results: 1h

    # --- context management ---
    context_soft_limit_tokens: int = 24_000
    tool_output_cap_chars: int = 12_000

    # --- execution sandbox & workspace (Phase 4, v0.5) ---
    workspace_dir: str = "~/.woyo/workspace"  # persistent cwd for code/shell
    sandbox_backend: str = "local"  # local | docker (docker = --network none)

    # --- sub-agents (v0.5) ---
    subagent_max_steps: int = 6
    subagent_max_time_s: float = 90.0
    subagent_max_cost_usd: float = 0.10
    subagent_max_search_calls: int = 4

    # --- chat frontends (v0.3) ---
    chat_password: str | None = None  # required to expose web chat beyond localhost
    chat_history_turns: int = 12  # transcript turns sent as context per message
    chat_daily_messages: int = 200  # per chat, per day
    direct_text_replies: bool = False  # chat: accept prose as the final answer
    chat_approval_timeout_s: int = 120  # telegram inline-button wait

    # --- browser automation (Phase 5, v0.9) ---
    browser_enabled: bool = True  # active when the `browser` extra is installed
    browser_allowlist: str = ""  # comma-separated domains; empty = public web
    browser_allow_private_hosts: bool = False  # intranet/test escape hatch
    browser_max_actions: int = 40  # actions per session (navigate/click/...)
    browser_session_ttl_s: int = 900  # hard session lifetime
    browser_idle_close_s: int = 300  # inactivity close
    browser_max_sessions: int = 3  # concurrent chromium contexts per process
    browser_no_sandbox: bool = False  # constrained hosts only (weakens isolation)

    # --- chat file transfer (v0.6) ---
    files_dir: str = "~/.woyo/files"  # durable inbox/outbox (synced on runners)
    chat_max_file_mb: int = 15  # inbound attachment cap (Bot API downloads max 20 MB)
    chat_max_send_file_mb: int = 45  # outbound upload cap (Bot API uploads max 50 MB)
    chat_extract_chars: int = 8_000  # extracted text inlined into the prompt per file
    vision_model: str | None = None  # ref for image-bearing calls, e.g. "openrouter:qwen/qwen3.8-27b:free"
    chat_asr_provider: str | None = None  # OpenAI-compat transcriptions endpoint, e.g. "groq"
    chat_asr_model: str = "whisper-large-v3"

    # --- scheduled jobs / follow-through (v0.8) ---
    job_poll_s: int = 30  # scheduler tick inside the chat frontends
    max_scheduled_jobs: int = 25  # active jobs per chat (limit)
    max_schedule_horizon_days: int = 30  # how far ahead a task may be scheduled
    job_max_attempts: int = 3  # host-rotation retries before a job fails out

    # --- persistence, tasks & memory (Phase 3, v0.4) ---
    db_path: str = "~/.woyo/woyo.sqlite3"  # tasks + memories (WAL)
    task_checkpoint: bool = True  # persist run state each step for resume
    task_stale_minutes: int = 10  # running rows older than this are recoverable
    memory_enabled: bool = True
    memory_embedder: str = "auto"  # auto | hash | openai
    memory_embed_model: str | None = None  # default depends on provider
    memory_recall_k: int = 4  # memories injected into a run's prompt
    memory_min_similarity: float = 0.04  # recall threshold (hash noise floor is ~0.00)
    memory_max_items: int = 5000  # LRU-ish cap (data minimization)
    memory_default_ttl_days: int = 180  # 0 = never expire

    # --- misc ---
    timezone: str = "UTC"
    sessions_dir: str = "~/.woyo/sessions"

    def db_file(self) -> Path:
        return Path(self.db_path).expanduser()

    # ------------------------------------------------------------------
    def resolved_api_key(self) -> str | None:
        """API key for the configured provider (WOYO_API_KEY wins, then preset env)."""
        return resolve_api_key(self.provider, self)

    def resolved_base_url(self) -> str | None:
        return resolve_base_url(self.provider, self)

    def model_for_role(self, role: Role) -> str:
        if role == "planner" and self.planner_model:
            return self.planner_model
        if role == "executor" and self.executor_model:
            return self.executor_model
        if role == "summarizer" and self.summarizer_model:
            return self.summarizer_model
        return self.model

    def sessions_path(self):
        from pathlib import Path

        return Path(self.sessions_dir).expanduser()


def _dotenv_lookup(names: list[str]) -> str | None:
    """Find a value for one of `names` in the CWD .env file (no override).

    pydantic-settings only loads WOYO_* fields into Settings; provider-native
    variables (GROQ_API_KEY, TAVILY_API_KEY, ...) live in the same .env and
    must be readable too. os.environ always wins over the file.
    """
    try:
        lines = Path(".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    wanted = {n.strip() for n in names}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in wanted:
            value = value.strip().strip("'\"")
            if value:
                return value
    return None


def env_value(*names: str) -> str | None:
    """First non-empty value among `names`: os.environ, then CWD .env."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return _dotenv_lookup(list(names))


def resolve_api_key(provider: str, settings: Settings) -> str | None:
    if provider == "mock":
        return "mock"
    if provider == "anthropic":
        if settings.api_key:
            return settings.api_key
        return env_value("ANTHROPIC_API_KEY")
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:  # unknown/custom provider name
        return settings.api_key or env_value(f"{provider.upper()}_API_KEY")
    if settings.api_key and provider == settings.provider:
        return settings.api_key
    key_env = preset.get("key_env")
    if key_env:
        names = [n.strip() for n in key_env.split(",")]
        for env_name in names:
            value = os.environ.get(env_name)
            if value:
                return value
        return _dotenv_lookup(names)
    return preset.get("key_default")


def resolve_base_url(provider: str, settings: Settings) -> str | None:
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return settings.base_url
    base = preset.get("base_url")
    if base == "WOYO_BASE_URL":
        return settings.base_url
    return base
