# Roadmap

Each phase has **exit criteria** — a phase is done when its criteria pass, not when its code exists. Dates are deliberately absent; correctness of ordering matters more than speed.

## Phase 1 — Agent core (MVP) ✅ (v0.1)

Core loop (plan → act → observe → adapt → verify → finish), tool system with permissions & approval gates, model abstraction (OpenAI-compatible + Anthropic + Mock) with role routing, budget enforcement (steps/time/tokens/cost/tool-calls/loop-detection), injection-hardened context handling, event stream + jsonl transcripts, CLI with live task view, `woyo doctor`, full mock-driven test suite.

- [x] Happy path: task → plan → tools → verified finish with sources
- [x] Recovery: tool failures produce typed observations; agent adapts; no crashes
- [x] Safety: approval gates, untrusted wrapping + flagging, SSRF-safe fetch, fail-safe defaults
- [x] Budgets: all limits enforced with honest partial results
- [x] Tests: whole suite green without network or API keys

## Phase 2 — Research depth ✅ (v0.2)

Robust `web_search`/`fetch_url` in the field: content extraction quality (trafilatura), result caching (URL → content, TTL), source tracking enforced in `finish`, comparison/cross-check patterns in executor prompts, per-task search budgets.

- [x] Agent completes a multi-source comparison task and cites accurate URLs (live Cohere run, v0.2 smoke test)
- [x] Repeated fetches of the same URL within a run hit cache (SQLite, 24h pages / 1h searches; test + live)
- [x] Bounded `crawl_site` tool: same-origin, page budget, politeness delay, cache-aware
- [x] Citation verification: claimed sources matched against URLs actually observed; invented ones dropped + flagged (`citation_flagged` event, `sources_dropped` in the report)
- [x] Cross-check guidance in executor prompts (two independent sources; report disagreements)
- [x] Per-task search budget (`WOYO_MAX_SEARCH_CALLS`, default 12)
- [ ] Extraction quality spot-check on 20 diverse pages (10 checked live: 8/8 reachable pages extracted well — 360 to 16,588 words; 1 rate-limited by target, 1 bad URL — rolling)

## Phase 3 — Persistence, tasks & memory

SQLite store; task state machine (pending/running/waiting_approval/paused/completed/failed/cancelled) with pause/resume/cancel/retry; conversation memory across turns; long-term memory with embeddings (SQLite + sqlite-vec); memory visibility & deletion (CLI `woyo memory` commands); data minimization (expiration, relevance).

- [ ] A task survives process restart and resumes correctly
- [ ] User can list/inspect/delete every memory item
- [ ] Memory measurably helps a follow-up task (repeat-question test)

## Phase 4 — Real code sandbox

Replace dev-grade `python_exec` with container isolation (Docker, or E2B as optional hosted path): no host network by default, resource caps, filesystem scoping, image pinning. Add file tools (workspace read/write/list) scoped to a task directory. Document a threat model for executed code.

- [ ] Sandbox escape attempts in tests stay contained (no host FS, no network unless granted)
- [ ] Agent analyzes a CSV dataset end-to-end and produces a report file

## Phase 5 — Browser automation

Playwright-driven browser tool following the browser-use accessibility-tree pattern: navigate/click/type/extract/screenshot, action budgets, domain allowlists, approval on submissions (the "final click"). Screenshots as observations for the model.

- [ ] Multi-page research workflow completes with sources
- [ ] A form-submission flow pauses for approval before the irreversible action
- [ ] Bot-wall / failure handling degrades to search+fetch gracefully

## Phase 6 — API server + web UI

FastAPI + SSE streaming the existing event bus; React UI (task view, plan, tool activity, approvals, history); auth (single-user first, multi-user later); tasks fully manageable from the UI. The CLI remains a first-class client.

- [ ] Same task runs identically via CLI and UI
- [ ] Approval requests delivered and answerable in the UI
- [ ] No secrets reachable from the client

## Phase 7 — Integrations

GitHub first (dogfood: woyo works on woyo — issues, PRs, releases), then Gmail / Calendar / Drive / Telegram per OAuth with per-integration grants, scoped tokens, and revocation. Each integration = Tool classes + a permission profile.

- [ ] An integration performs an external write only after explicit approval
- [ ] Tokens revocable per integration; least-scope documented

## Phase 8 — Advanced autonomy & evaluation

Eval harness (success rate, cost, latency, recovery rate over a task corpus); adaptive model routing by task class; optional planner/executor/critic multi-agent mode behind the same interface; prompt/regression testing in CI; cost dashboards.

- [ ] Eval corpus runs in CI with tracked metrics
- [ ] Routing policy demonstrably cuts cost without cutting success rate

---

### Cost posture (what's free, what isn't)

| Capability | Free path | Paid path (optional) |
|---|---|---|
| Models | Ollama (local); Google AI Studio / Groq / OpenRouter free tiers | Frontier models for hard reasoning, per-role routing keeps usage low |
| Search | DuckDuckGo (keyless) | Tavily 1k/mo free → $30/mo beyond; Brave |
| Extraction | trafilatura (local) | Firecrawl/Crawl4AI hosted if needed |
| Memory/storage | Files + SQLite + sqlite-vec | — (no paid vector DB needed) |
| Code sandbox | Docker (local) | E2B hosted convenience |
| Hosting | localhost / self-host | VPS for the Phase 6 UI |

---

## Provisioned credentials → phase mapping (Sept 2026)

Keys the owner already holds, and where they enter the roadmap:

| Credential | Phase | Purpose |
|---|---|---|
| Groq / Gemini / OpenRouter / Mistral / Cohere / xAI / HuggingFace keys | now | model providers (see README provider matrix) |
| `GITHUB_TOKEN` | Phase 7 | GitHub integration (issues, PRs, repo ops) — first integration target |
| Telegram bot token | Phase 7 | approvals & notifications channel for headless runs |
| Supabase (URL + service key) | Phase 3+ (optional) | hosted alternative to SQLite/sqlite-vec for memory & task queue; self-host stays default |
| Cloudflare API token | Phase 6 (optional) | web UI hosting / R2 artifact storage |
| Vercel token | Phase 6 (optional) | alternative web UI hosting |

Rotation reminder: these keys were shared in plaintext during setup — rotate them before relying on any of them in production.
