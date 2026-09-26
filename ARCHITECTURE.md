# woyo — Architecture Blueprint

This document is the engineering blueprint for woyo: what we're building, why the architecture looks the way it does, and how it grows over time. It is a living document — each phase updates it.

- Vision & product framing: [README.md](README.md)
- Decision records & trade-offs: [docs/DECISIONS.md](docs/DECISIONS.md)
- Threat model & security: [docs/SECURITY.md](docs/SECURITY.md)
- Phased plan with exit criteria: [docs/ROADMAP.md](docs/ROADMAP.md)

---

## 1. What "general-purpose agent" means here

woyo is a **runtime for goal-directed work**, not a chat wrapper. The contract with the user:

```
goal in ─▶ understand ─▶ plan ─▶ act ─▶ observe ─▶ evaluate ─▶ (replan)* ─▶ verify ─▶ report out
```

Three properties distinguish this from a chatbot with tools bolted on:

1. **Deliberation is explicit.** A planner produces a `TaskPlan` (goal, success criteria, steps) before any action, and the plan is visible to the user and re-evaluable by the agent.
2. **Feedback is first-class.** Every action produces an observation with a *typed outcome* (`ok`, or an `ErrorKind`: transient / tool failure / invalid input / missing info / needs approval / impossible). The executor reasons over these outcomes instead of crashing.
3. **Termination is engineered.** Budgets (steps, time, tokens, cost, tool calls), loop detection, and an explicit `finish` tool with self-verification. The agent stops — honestly — instead of looping forever or hallucinating completion.

## 2. Top-level architecture

```
┌─────────────────────────── Presentation ───────────────────────────┐
│  CLI (rich live view)        [P6] FastAPI + SSE + Web UI           │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │  events ▲   commands ▼
┌───────────────────────────── Agent core ───────────────────────────┐
│  AgentRunner (loop.py)                                             │
│    ├─ Planner            LLM role: task ─▶ TaskPlan (JSON)         │
│    ├─ Executor           LLM role: native tool calling, ReAct-ish  │
│    ├─ Replan triggers    failure pressure ─▶ strategy change       │
│    ├─ Approval gates     consequential actions pause for the user  │
│    ├─ RunLimits          steps/time/tokens/cost/tool-calls/loops   │
│    └─ ContextManager     compaction of old tool output             │
│                                                                    │
│  ModelRouter            role ─▶ (provider, model); usage & cost    │
│    ├─ OpenAI-compatible provider  (OpenAI/Groq/OpenRouter/         │
│    │                               DeepSeek/Ollama/vLLM/custom)    │
│    ├─ Anthropic provider                                  [P1.1]  │
│    └─ Mock provider               deterministic tests             │
│                                                                    │
│  ToolRegistry                                                       │
│    ├─ Tool protocol: typed args (pydantic) + JSON schema           │
│    ├─ Permission: read_only / sandboxed / writes_external /        │
│    │              destructive  ─▶ drives approval gates            │
│    ├─ Execution: validate → timeout → run → cap → wrap             │
│    └─ Builtins: web_search, fetch_url, calculate, now,             │
│                 ask_user, finish, python_exec(off by default)      │
│                                                                    │
│  EventBus              structured events ─▶ jsonl + subscribers    │
│  SessionStore          transcripts (jsonl) under ~/.woyo           │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                     ┌── Future (see ROADMAP) ──┐
                     │ P3 SQLite + memory tiers │
                     │ P4 Docker code sandbox   │
                     │ P5 Playwright browser    │
                     └──────────────────────────┘
```

**Dependency rule:** everything points inward. Tools don't know about the loop; the loop doesn't know about the CLI; models don't know about tools (they receive tool *schemas*). Any box can be replaced without touching the others.

## 3. The agent loop (v0.1, implemented)

1. **Plan.** `Planner` (an LLM call with no tools) turns the task + tool catalog into a `TaskPlan`. Parse failures degrade gracefully to a single-step fallback plan — a broken planner never blocks execution.
2. **Execute.** The executor LLM sees: hardened system prompt (role, discipline, security rules, quality rules), the task, the rendered plan, and the running history. It responds with **native tool calls** (provider function-calling — never free-text JSON, which is both unreliable and injection-prone).
3. **Gate.** Each requested tool call is checked: does it exist, do its arguments validate, does its permission require user approval? Destructive/external actions pause the run and surface an approval request (denied by default when no human is reachable — fail safe).
4. **Act & observe.** The registry executes with a timeout, caps output size, and wraps external content in `<untrusted>` blocks (see §6). Observations are appended as tool messages.
5. **Adapt.** Consecutive failures of the same tool, or repeated identical calls, inject a *replan notice* into the conversation ("change strategy"). The plan is guidance; the executor may deviate and must say so.
6. **Finish.** The `finish` tool is the only exit for success: it requires a summary, a `verified` flag the model may only set if it checked its success criteria, `sources` (URLs actually used), and `open_questions`. Budget exhaustion produces an honest partial result, never a fabricated one.

**Why a single-loop executor (not a graph/multi-agent system) for v0.1?** Simplicity is a feature. 90%+ of useful tasks fit one competent loop with a plan; multi-agent orchestration adds failure modes, cost, and debugging pain that pay off only for genuinely parallelizable work (Phase 8 evaluates it). See docs/DECISIONS.md#ADR-2.

## 4. Tool system

Every tool is a small, self-describing plugin:

```python
class Tool(ABC):
    name: str; description: str; permission: Permission; timeout_s: float
    Args: type[BaseModel]                      # typed arguments
    async def run(self, args) -> ToolResult    # never raises
```

Lifecycle: `register → expose schema → model calls → validate args → (approval?) → timeout-boxed run → capped + wrapped observation`.

- **Typed args**: pydantic models generate the JSON schema handed to the LLM *and* validate what comes back. Invalid arguments become an `invalid_input` observation the model can self-correct from — not a crash.
- **Permissions drive policy, not configuration scattered everywhere**: `read_only` runs freely; `sandboxed` runs when enabled; `writes_external`/`destructive` require explicit user approval (configurable).
- **Errors are data**: `ToolResult(ok=False, error_kind=..., error_message=...)` flows back into the conversation with machine-readable structure.

Adding the Phase 5 browser or Phase 7 integrations (GitHub, email, calendar) means writing new `Tool` classes — zero changes to the core.

## 5. Model layer & routing

- **Canonical message format** (≈ OpenAI shape) everywhere inside the core; each provider converts at the boundary. Anthropic's system/tool_result differences are contained in `anthropic_provider.py`.
- **Model refs** like `groq:llama-3.3-70b-versatile` or bare model names (default provider) allow per-role assignment: a cheap/fast model for planning and summarization, a strong model for execution — or one model for everything.
- **Usage & cost accounting** per model per run, with a price table for common models and safe-zero + warning for unknowns. Feeds the cost budget and the run report.
- **Retries** with exponential backoff on transient provider errors (429/5xx/timeouts); never on auth or argument errors.
- The **Mock provider** makes the whole system testable end-to-end without keys or network — this is what keeps the core honest.

**Deliberately not adopted for v0.1:** LiteLLM (59k⭐, capable, but heavy and gateway-oriented), LangGraph/PydanticAI as foundations (excellent projects — borrowed ideas from both; rationale in docs/DECISIONS.md#ADR-1). We own ~300 lines of provider code; the moment that stops being enough, swapping in an off-the-shelf gateway is a contained change behind the same interface.

## 6. Security model (summary — full threat model in docs/SECURITY.md)

- **Prompt injection (OWASP LLM01)**: user/system instructions and external data live in separate channels. All tool output from the outside world arrives wrapped: `<untrusted source="web:example.com">…</untrusted>`, with an explicit system-prompt contract that such content is data, never instructions, plus a best-effort detector that flags instruction-shaped payloads in observations. Defense is layered and *tested* (tests/test_safety.py), with the honest caveat that no LLM system is injection-proof — which is why approval gates and read-only defaults exist.
- **SSRF**: `fetch_url` resolves DNS itself, rejects private/loopback/link-local targets, allows only http/https on standard ports, caps size, follows ≤5 redirects with re-validation per hop.
- **Sandboxing**: `python_exec` is off by default. v0.1 offers dev-grade isolation (isolated interpreter, rlimits, timeout, temp cwd) and is *documented as not a security boundary*; Phase 4 replaces it with proper container isolation.
- **Secrets**: environment-only, never in transcripts, never in the repo; arguments are digest-logged by default.
- **Fail-safe defaults**: no approval channel reachable → consequential actions are denied, not silently run.

## 7. Memory (v0.1 → Phase 3)

v0.1 ships **working memory** (the conversation with compaction: oldest bulky tool outputs are collapsed when the estimated context exceeds a soft limit, preserving protocol validity) and **session transcripts** (jsonl). Phase 3 introduces the tiered design — task state (pause/resume), conversation memory, long-term memory with embeddings on SQLite + sqlite-vec — with explicit user visibility and deletion, data minimization, and expiration. No unbounded hoarding.

## 8. Observability

Every run emits structured events (`run_started`, `plan_created`, `llm_call`, `tool_call`, `tool_result`, `approval_requested`, `injection_flagged`, `limit_hit`, `run_finished`, …) to subscribers and a jsonl sink. The CLI renders them live; the Phase 6 web UI consumes the same stream over SSE; developers grep them. Sensitive values (full secrets, raw long args) stay out of events by design.

## 9. Data model (Phase 3 preview)

SQLite tables: `users`, `tasks` (state machine: pending → running → waiting_approval → paused → completed | failed | cancelled), `task_events`, `messages`, `memories` (kind, content, embedding, expires_at, source), `tool_grants` (per-tool permission decisions), `usage_ledger`. Designed, not yet built — v0.1 needs none of it, which is the point of the phased plan.

## 10. What we explicitly are not doing (yet)

- No multi-agent orchestration (Phase 8 evaluates against real workloads)
- No always-on background autonomy — every run is user-initiated and budgeted
- No unrestricted shell — code execution is sandboxed or absent
- No framework lock-in — the core owns the loop
