# woyo

**A general-purpose AI agent you own.** Give woyo a goal — not a procedure. It understands the task, makes a plan, picks the right tools, acts, observes what happened, adapts when things break, and reports back with sources and honest uncertainty. Human approval is built in wherever actions have consequences.

> 🌐 Project site: **https://bessghiermohamed.github.io/woyo/**

> *"Find volunteering opportunities that match my profile, check their requirements, and tell me which ones I'm actually eligible for."* — this is the level woyo is built for. You should not have to specify the websites, the queries, or the steps.

```
User goal ─▶ Understand ─▶ Plan ─▶ Act (tools) ─▶ Observe ─▶ Evaluate
                 ▲                                            │
                 └──────────── Replan on failure ◀────────────┘
                              ──▶ Verify ─▶ Report (with sources)
```

## Why woyo exists

Most "AI agents" are chatbots with an agent-themed UI. woyo is a from-scratch agent **runtime** designed around the things that actually make agents trustworthy:

| Principle | What it means in code |
|---|---|
| **Real loop, not a script** | Plan → act → observe → adapt → verify, with a genuine state machine and budget enforcement (`src/woyo/agent/loop.py`) |
| **Tools are plugins** | Standardized `Tool` protocol with typed args, permissions, timeouts — register a new tool and the agent discovers it (`src/woyo/tools/`) |
| **No vendor lock-in** | Model providers are swappable per role (planner / executor / summarizer) — OpenAI, Anthropic, Groq, Gemini, OpenRouter, Mistral, xAI (Grok), Cohere, HuggingFace, DeepSeek, Ollama, any OpenAI-compatible endpoint (`src/woyo/models/`) |
| **Security is structural** | External content is wrapped as `<untrusted>` data, never instructions; approval gates for consequential actions; SSRF-safe fetching; budget caps (docs/SECURITY.md) |
| **Observable by design** | Every run emits structured events (tool calls, durations, token usage, cost estimates) — the foundation for the upcoming UI |
| **Stops when it should** | Step / time / token / cost / tool-call limits, loop detection, and an honest `verified: false` when it couldn't check its own work |

## Status

**Phase 1 (MVP) — done (v0.1).** Agent core loop, planner, tool system with permissions, model routing, budget enforcement, injection defenses, CLI with live task view, full mock-driven test suite.

**Phase 2 (research depth) — done (v0.2).** SQLite response cache (24h pages / 1h searches) so repeated fetches never re-hit the network, `crawl_site` bounded same-origin crawler, **citation verification** (claimed sources are matched against URLs actually observed in the run — invented citations are dropped and flagged), cross-check guidance in executor prompts, per-task search budgets. Plus a project site on GitHub Pages.

**Chat frontends — done (v0.3).** Talk to woyo from your phone: a **Telegram bot** and a **mobile-first web chat**, both served by the same agent core (transcript memory, per-message budgets, citation-checked answers). Runs anywhere Python runs — including a free Hugging Face Space.

Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) — next up: persistent tasks & memory (Phase 3), real code sandbox (Phase 4), browser automation (Phase 5), web UI (Phase 6).

## Quickstart

```bash
git clone https://github.com/bessghiermohamed/woyo
cd woyo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add ONE model key (e.g. OpenAI, Groq, OpenRouter)
woyo doctor                 # verify your setup
```

Run a task:

```bash
woyo run "What are the three most recent Python releases and what are the headline changes? Cite sources."
```

You'll see the live plan, each tool call as it happens, token/cost usage — then a final answer with sources. Interactive mode:

```bash
woyo chat
```

## Chat with woyo (v0.3)

Two frontends turn woyo into a phone-reachable assistant — no computer needed on your side:

```bash
# Telegram: talk to @BotFather → /newbot → copy the token into .env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
woyo telegram            # the first /start claims the bot; commands: /new /status /help

# Web: a self-contained chat page (works great on phones)
WOYO_CHAT_PASSWORD=let-me-in
woyo web --host 0.0.0.0 --port 7860   # passcode-gated, rate-limited
```

Both share the same core: rolling transcript memory (survives restarts via `~/.woyo/chats/`), tighter per-message budgets, tools on demand (search / fetch / calculate), and answers with verified sources. A ready-to-copy deployment for a **free Hugging Face Space** lives in [`deploy/hf-space/`](deploy/hf-space/) — web chat plus the Telegram poller in one container.

### Free-tier friendly

woyo is designed to run at $0 for typical use:

- **Models**: Google AI Studio (Gemini Flash, generous free tier), Groq, OpenRouter free models (`:free` suffix), Mistral, Cohere trial keys, HuggingFace inference credits — or fully local via Ollama. Any OpenAI-compatible endpoint works.
- **Search**: Tavily free tier (1,000 credits/mo) or Brave; DuckDuckGo needs no key at all (`pip install -e ".[search]"`).
- **Storage**: plain files + SQLite. No paid vector DB required.

<details>
<summary><strong>Provider matrix (verified Sept 2026)</strong></summary>

| Provider | Env var(s) | Example model | Cost |
|---|---|---|---|
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | paid |
| groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free tier |
| gemini | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `gemini-2.5-flash` | free tier |
| openrouter | `OPENROUTER_API_KEY` | `qwen/qwen3.8-27b:free` | 17 free models |
| mistral | `MISTRAL_API_KEY` | `mistral-small-latest` | free tier |
| xai | `XAI_API_KEY` | `grok-4-fast-non-reasoning` | paid credits |
| cohere | `COHERE_API_KEY` | `command-a-03-2025` | trial key |
| huggingface | `HF_TOKEN` | `openai/gpt-oss-120b` | included credits |
| deepseek | `DEEPSEEK_API_KEY` | `deepseek-chat` | cheap |
| anthropic | `ANTHROPIC_API_KEY` | `claude-haiku` (native SDK) | paid |
| ollama | — | `qwen3:8b` | free, local |
| g4f | — | any | **experimental** — see below |
| custom | `WOYO_API_KEY` + `WOYO_BASE_URL` | any OpenAI-compatible endpoint | varies |

**GPT4Free (`g4f`)**: wired as an opt-in extra (`pip install "woyo[g4f]"`, `WOYO_PROVIDER=g4f`) for experimentation. Tested Sept 2026 from a clean Linux box: the default provider chain failed (some routes need a local Chrome, others are IP-blocked or paywalled). It is unreliable by nature and sends your prompts to unvetted third parties — never use it for anything sensitive. Your sanctioned free tiers above are strictly better. See ADR-9 in [docs/DECISIONS.md](docs/DECISIONS.md).

</details>

## Built-in tools (v0.2)

| Tool | What it does | Permission |
|---|---|---|
| `web_search` | Web search via Tavily / Brave / DuckDuckGo (auto-detected, results cached 1h) | read-only |
| `fetch_url` | Fetch a page, extract main content (SSRF-hardened, cached 24h) | read-only |
| `crawl_site` | Bounded same-origin crawl: several pages from one site in one call | read-only |
| `calculate` | Safe math expressions | read-only |
| `now` | Current date/time in your timezone | read-only |
| `ask_user` | Ask the human a question when blocked | read-only |
| `finish` | End the task with summary, verification, **citation-checked** sources | control |
| `python_exec` | Sandboxed Python (disabled by default; dev-grade sandboxing in v0.2 — see SECURITY.md) | sandboxed |

Adding your own tool takes ~20 lines:

```python
from pydantic import BaseModel
from woyo.tools import Tool, ToolResult, Permission

class WeatherArgs(BaseModel):
    city: str

class WeatherTool(Tool):
    name, description, permission = "get_weather", "Current weather for a city", Permission.READ_ONLY
    Args = WeatherArgs

    async def run(self, args: WeatherArgs) -> ToolResult:
        ...  # call an API
        return ToolResult.ok(f"{args.city}: 21C, clear")
```

Register it, and the agent can use it on the next task. No core changes.

## Architecture (short version)

```
CLI / (future) Web UI
        │
   Agent loop  ── Planner (LLM role)          ── TaskPlan
        │       ── Executor (LLM role)        ── tool calls via native function calling
        │       ── ModelRouter                ── provider + model per role, usage/cost accounting
        │       ── ContextManager             ── compaction when context grows
        │
   ToolRegistry ── typed args, permissions, timeouts, approval gates, untrusted wrapping
        │
   EventBus ── structured events (jsonl) ── observability & future UI
```

Full blueprint: [ARCHITECTURE.md](ARCHITECTURE.md) · key decisions & trade-offs: [docs/DECISIONS.md](docs/DECISIONS.md) · threat model: [docs/SECURITY.md](docs/SECURITY.md)

## Development

```bash
pip install -e ".[dev]"
pytest                  # full suite, no network or API keys needed (mock-driven)
ruff check src tests
```

The test suite runs the *entire* agent loop against a scripted mock model provider — happy path, failures, prompt-injection attempts, budget exhaustion, loop detection, approval gates — so the core behavior is verified without spending a cent.

## Contributing

Issues and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first; security reports should follow [docs/SECURITY.md](docs/SECURITY.md).

## License

[MIT](LICENSE)
