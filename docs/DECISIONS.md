# Decision Records (ADRs)

Architecture Decision Records for woyo. Each records a choice, the alternatives considered, and the reasoning. Current as of September 2026.

---

## ADR-1: Build the agent core from scratch (thin custom core) instead of adopting an agent framework

**Status:** Accepted (v0.1). Revisit at Phase 4/5 boundaries.

**Alternatives considered** (activity verified on GitHub, Sept 2026):

| Option | Stars | State | Assessment |
|---|---|---|---|
| LangGraph | 42.3k | very active, MIT | Powerful stateful graphs + checkpointing; steep curve, opinionated runtime, and our differentiators (approval gates mid-batch, custom compaction, injection channeling, event stream) end up fighting the graph abstraction |
| OpenAI Agents SDK | 29.7k | very active, 21 open issues, MIT | Cleanest mainstream design; OpenAI-shaped worldview, handoffs/guardrails we'd partly reimplement anyway |
| Microsoft Agent Framework | 13.8k | active (AutoGen + Semantic Kernel merger) | Enterprise-leaning, heavy for a personal agent |
| CrewAI | 59k | very active, MIT | Role-play/multi-agent oriented — wrong shape for a single supervised loop |
| PydanticAI | 20.2k | very active, MIT | *Closest to our taste*: typed tools, multi-provider, light. Chosen ideas to borrow |
| **Custom thin core** | — | — | **Chosen** |

**Reasoning:**

1. **The loop is the product.** The vision's core requirements — approval gates, budget governance, injection-resistant context handling, replanning pressure, honest termination — are all *cross-cutting concerns inside the loop itself*. Owning ~1.5k lines of core beats debugging a framework's abstractions at every one of those points.
2. **Churn risk.** Agent framework APIs have historically rewritten themselves (LangChain → LCEL → LangGraph). A personal, long-lived project should not inherit that.
3. **Learning & auditability.** Every line is readable in an afternoon; security review is actually possible.
4. **Exit doors stay open.** Provider access is the most churn-prone layer; our `ModelProvider` interface can wrap LiteLLM (59.6k⭐) later if provider coverage outgrows our two adapters. The loop itself can host a LangGraph sub-graph behind a `Tool` if Phase 8 demands it.

**Consequences:** we implement retries, tool-schema generation, and two protocol adapters ourselves (~300 lines, fully tested). Accepted.

---

## ADR-2: Single supervised loop + planner, not multi-agent, for v0.1

Multi-agent systems multiply cost and failure modes; most tasks in scope are serial and benefit from one coherent context. A planner/executor split captures 80% of the benefit (deliberation separated from action) with 10% of the complexity. Revisit with real Phase 8 workloads and an eval harness.

## ADR-3: Native provider tool-calling, never prompt-parsed JSON tools

Function-calling APIs return tool invocations through a structured channel, which is more reliable and shrinks the injection surface (arguments still get validated). Free-text "call this tool as JSON" protocols remain available as a compatibility mode for endpoints without tool support, but are not the default. (The `dev/zai_gateway.mjs` utility demonstrates this emulation for dev/testing.)

## ADR-4: Python 3.11+, asyncio-first

The agent's real work is I/O (LLM calls, HTTP, later Playwright). Async-first keeps concurrency natural when Phase 5 adds parallel page actions. Python owns the best tooling ecosystem for this project (httpx, pydantic, Playwright, ddgs, trafilatura, document parsers).

## ADR-5: Files + JSONL now, SQLite at Phase 3, Postgres only if multi-user demands it

v0.1's persistence needs are transcripts and events — append-only files are inspectable with `cat` and impossible to corrupt with a bad migration. SQLite (with sqlite-vec for embeddings) covers single-user memory without running a database server. Postgres/pgvector is the Phase 6+ path *if* the web UI goes multi-user.

## ADR-6: Search backends are adapters behind one tool

Tavily (free 1,000 credits/mo), Brave (free tier), DuckDuckGo (keyless via `ddgs`) are interchangeable behind `web_search`. The agent never knows which is active — cost/availability becomes a config decision, not an architecture one.

## ADR-7: Approval policy is permission-driven and fail-safe

Tools declare `Permission`; the runtime decides who must approve. With no interactive channel available, consequential actions are **denied by default**. This is deliberately conservative: an agent that under-acts can be re-run with approvals; an agent that over-acts cannot be un-run.

## ADR-8: CLI before web UI

The vision demands a rich UI eventually — but UIs hide agent behavior behind polish. Building the CLI live-view first (plan, tool calls, budget ticker) forces the event stream to be complete and useful, which is exactly the contract the Phase 6 web UI will consume. The UI will be *another subscriber*, not a rewrite.

## ADR-9: GPT4Free is an opt-in experimental extra, never a default provider

**Context.** The user asked to try GPT4Free (`g4f`) as a free model source alongside sanctioned free tiers.

**Decision.** woyo wires g4f as an optional dependency (`pip install "woyo[g4f]"`, `WOYO_PROVIDER=g4f`) behind a thin adapter (`src/woyo/models/g4f_provider.py`) that reuses the OpenAI-compat conversion. It is excluded from default dependencies and defaults.

**Why not adopt it.**
- *Verified failure mode (Sept 2026 live test from a clean Linux box)*: the default provider chain failed — some routes require a local Chrome/Chromium, others are IP-blocked (403) or paywalled. Reliability is inherent to how g4f sources endpoints.
- *ToS exposure*: several upstream endpoints do not sanction automated access; using them at scale risks account bans and legal ambiguity.
- *Data safety*: prompts travel to unvetted third parties with no SLA — categorically unsafe for anything confidential.
- *License*: g4f's license terms are nonstandard for embedding; keeping it a runtime-optional extra (never vendored, never a hard dependency) avoids contaminating woyo's MIT surface.

**The alternative is strictly better.** Sanctioned free tiers — Gemini Flash, Groq, OpenRouter `:free` models (17 available), Mistral, Cohere trial keys, HuggingFace included credits, local Ollama — are free, legitimate, and stable. g4f stays available for experimentation and for users who accept the trade-offs knowingly.
