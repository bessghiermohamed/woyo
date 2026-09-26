# woyo

**A general-purpose AI agent you own.** Give woyo a goal — not a procedure. It understands the task, makes a plan, picks the right tools, acts, observes what happened, adapts when things break, and reports back with sources and honest uncertainty. Human approval is built in wherever actions have consequences.

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
| **No vendor lock-in** | Model providers are swappable per role (planner / executor / summarizer) — OpenAI, Anthropic, Groq, OpenRouter, DeepSeek, Ollama, any OpenAI-compatible endpoint (`src/woyo/models/`) |
| **Security is structural** | External content is wrapped as `<untrusted>` data, never instructions; approval gates for consequential actions; SSRF-safe fetching; budget caps (docs/SECURITY.md) |
| **Observable by design** | Every run emits structured events (tool calls, durations, token usage, cost estimates) — the foundation for the upcoming UI |
| **Stops when it should** | Step / time / token / cost / tool-call limits, loop detection, and an honest `verified: false` when it couldn't check its own work |

## Status

**Phase 1 (MVP) — working.** Agent core loop, planner, tool system with permissions, model routing, budget enforcement, injection defenses, CLI with live task view, full mock-driven test suite.

Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) — next up: deep research (Phase 2), persistent tasks & memory (Phase 3), real code sandbox (Phase 4), browser automation (Phase 5), web UI (Phase 6).

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

### Free-tier friendly

woyo is designed to run at $0 for typical use:

- **Models**: Google AI Studio (Gemini Flash, generous free tier), Groq, OpenRouter free models, or fully local via Ollama — any OpenAI-compatible endpoint works.
- **Search**: Tavily free tier (1,000 credits/mo) or Brave; DuckDuckGo needs no key at all (`pip install -e ".[search]"`).
- **Storage**: plain files + SQLite. No paid vector DB required.

## Built-in tools (v0.1)

| Tool | What it does | Permission |
|---|---|---|
| `web_search` | Web search via Tavily / Brave / DuckDuckGo (auto-detected) | read-only |
| `fetch_url` | Fetch a page, extract main content (SSRF-hardened) | read-only |
| `calculate` | Safe math expressions | read-only |
| `now` | Current date/time in your timezone | read-only |
| `ask_user` | Ask the human a question when blocked | read-only |
| `finish` | End the task with summary, verification, sources | control |
| `python_exec` | Sandboxed Python (disabled by default; dev-grade sandboxing in v0.1 — see SECURITY.md) | sandboxed |

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
