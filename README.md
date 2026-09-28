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

**Persistence, tasks & memory — done (v0.4).** A single SQLite store (WAL) backs a **resumable task queue** — the loop checkpoints itself after every step, so a SIGKILL mid-task loses nothing; `woyo tasks run --id N --recover` continues exactly where it died (verified live). **Long-term memory** recalls relevant facts into every prompt and is fully auditable: `woyo memory list|show|delete|prune`. Offline-first embeddings (zero new dependencies), TTLs and caps for data minimization.

**Execution, sub-agents & pressable approvals — done (v0.5).** The agent can now *do*, not just research: **`python_exec`** runs code in a sandbox (isolated interpreter, rlimits, scrubbed env so secrets never leak to child processes, optional `--network none` docker tier) with a **persistent workspace**; **`shell_exec`** is a real terminal, denied twice (config flag **and** per-call approval); **workspace file tools** (`read_file`/`write_file`/`list_dir`) with traversal protection; **`spawn_agent`** delegates subtasks to sub-agents (depth-1 by construction, one shared budget, citations still verified by the parent). In Telegram, approvals are **inline Approve/Deny buttons** — the bot shows the exact command and waits for your press (timeout = deny). Verified live end-to-end on Cohere: file → code → correct arithmetic, and sub-agent research delegation.

**Files in, files out — done (v0.6).** The chat became a real file channel. **Send the bot anything**: text/code/data files (contents inlined into its reasoning), PDFs (text extracted, page- and time-capped), archives (listed, never auto-extracted — zip bombs stay inert), photos (**vision** via free OpenRouter models, tried in order with an honest text fallback that never pretends to have seen the image), voice notes (optional ASR). Every extracted byte is wrapped as untrusted data. Files persist across runner rotations in a size-capped durable store. **Ask it to create files**: `write_file`/`python_exec` then **`send_file`** — the document lands right in your chat (verified live on Cohere: "create a CSV and send it" → `squares.csv` delivered with a caption in 11 s). The tool has exactly one possible destination: the chat asking for it.

Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) — next up: browser automation (Phase 5), web UI (Phase 6), integrations (Phase 7).

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

Both share the same core: rolling transcript memory (survives restarts via `~/.woyo/chats/`), tighter per-message budgets, tools on demand (search / fetch / calculate), and answers with verified sources.

**Hosting at $0:** [`deploy/gha-telegram/`](deploy/gha-telegram/) runs the Telegram bot on **GitHub Actions** (public repos get unlimited minutes) — a cron workflow keeps a runner alive ~24/7 and syncs conversation state to a private repo between runs. For a container deployment (needs HF PRO since 2026), [`deploy/hf-space/`](deploy/hf-space/) runs web chat plus the Telegram poller in one Space.

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

## Built-in tools (v0.6)

| Tool | What it does | Permission |
|---|---|---|
| `web_search` | Web search via Tavily / Brave / DuckDuckGo (auto-detected, results cached 1h) | read-only |
| `fetch_url` | Fetch a page, extract main content (SSRF-hardened, cached 24h) | read-only |
| `crawl_site` | Bounded same-origin crawl: several pages from one site in one call | read-only |
| `memory_save` | Store a durable fact/preference/note for future sessions (v0.4) | read-only (local) |
| `memory_search` | Semantic search over long-term memory (v0.4) | read-only |
| `calculate` | Safe math expressions | read-only |
| `now` | Current date/time in your timezone | read-only |
| `ask_user` | Ask the human a question when blocked | read-only |
| `finish` | End the task with summary, verification, **citation-checked** sources | control |
| `python_exec` | Run Python in the sandbox (isolated interpreter, rlimits, scrubbed env, persistent workspace; docker tier available) — off by default | sandboxed |
| `shell_exec` | Real terminal in the workspace — needs `WOYO_ENABLE_SHELL=true` **and** per-call approval (inline buttons in chat) | writes_external |
| `read_file` / `write_file` / `list_dir` | Workspace files with path-traversal protection (v0.5) | read-only / sandboxed |
| `spawn_agent` | Delegate a subtask to a fresh sub-agent (own plan + budgets, depth-1, shared cost account, citations re-verified by the parent) (v0.5) | read-only |
| `send_file` | Deliver a workspace file to the user's chat as a document (v0.6; chat frontends only — destination is fixed to the requesting chat) | sandboxed |

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

## Tasks that survive restarts + memory you can audit (v0.4)

Long tasks are queue-backed and **resumable**: the agent checkpoints its whole
loop state (conversation, budgets spent, elapsed time) after every step into
SQLite. Kill the process mid-run — the next `woyo tasks run --id N --recover`
picks up exactly where it died, budget accounting included.

```bash
woyo tasks add "Research X and cite official sources"   # queue it
woyo tasks run --id 1 --yes                               # run it (live event view)
woyo tasks pause 1        # ask a running task to pause (checkpointed)
woyo tasks cancel 1       # or cancel it
woyo tasks retry 1        # re-queue a finished task from scratch
woyo tasks list           # pending / running / waiting_approval / paused / ...
```

Long-term memory works across sessions and processes. Relevant memories are
auto-recalled into each run's prompt (framed as untrusted hints, never as
sources), and the agent can store what it learns with `memory_save`.
Everything is yours to audit — an assistant that remembers things you can't
see or delete is a liability:

```bash
woyo memory list                        # what do you know about me?
woyo memory search "python release"     # semantic search
woyo memory show 3                      # full record: source, expiry, accesses
woyo memory delete 3                    # your data, your call
woyo memory prune                       # purge expired (data minimization)
```

Memories default to a 180-day TTL and a 5,000-item LRU cap. Embeddings are
offline-first (deterministic hashing, zero dependencies); providers with an
OpenAI-compatible `/embeddings` route upgrade automatically. See ADR-11 in
[docs/DECISIONS.md](docs/DECISIONS.md).

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
