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

## Chat frontends ✅ (v0.3, pulled forward from Phase 7)

The user has no computer — a phone-reachable chat became the priority
(ADR-10). Delivered ahead of schedule:

- [x] Telegram bot: long-polling, zero new deps, owner-claim access control, /new /status /help
- [x] Mobile-first web chat: stdlib server + one embedded page, passcode-gated, rate-limited
- [x] Shared ChatSession core: transcript memory persisted across restarts, per-message budgets
- [x] Citation-checked answers in chat (live Cohere run: research turn with verified sources)
- [x] Free hosting: `deploy/gha-telegram/` — GitHub Actions runner + private-repo state sync (HF Space recipe kept for PRO accounts)
- [x] Telegram inline approval buttons (v0.5 — makes write-tools usable in chat)
- [x] File channel both ways (v0.6, ADR-13): inbound attachments ingested (text/PDF/zip-listing inlined as untrusted data, photos via vision models, voice via optional ASR, caps everywhere), `send_file` delivers workspace files to the requesting chat, durable size-capped store synced across runner rotations — *verified live on Cohere: "create a CSV and send it" → squares.csv delivered, 11 s*

## Phase 3 — Persistence, tasks & memory ✅ (v0.4.0)

SQLite store; task state machine (pending/running/waiting_approval/paused/completed/failed/cancelled) with pause/resume/cancel/retry; conversation memory across turns; long-term memory with embeddings (SQLite, brute-force KNN now — sqlite-vec past ~10k items, see ADR-11); memory visibility & deletion (CLI `woyo memory` commands); data minimization (expiration, relevance).

- [x] A task survives process restart and resumes correctly — *verified live: SIGKILL mid-run (3 steps done, 27.4s elapsed checkpointed), `--recover` resumed from step 3 and completed with verified sources*
- [x] User can list/inspect/delete every memory item — *`woyo memory list|show|delete|prune|stats`, exercised live*
- [x] Memory measurably helps a follow-up task (repeat-question test) — *verified live across processes: fact stored via `memory_save`, fresh process answered "what's my cat's name?" with 0 web searches; recall threshold calibrated on measured similarities*

## Phase 4 — Real code sandbox ✅ (v0.5.0)

Replace dev-grade `python_exec` with container isolation (Docker, or E2B as optional hosted path): no host network by default, resource caps, filesystem scoping, image pinning. Add file tools (workspace read/write/list) scoped to a task directory. Document a threat model for executed code.

Shipped in v0.5.0: two-tier sandbox — `local` backend (isolated interpreter `-I`, POSIX rlimits, **scrubbed env** so runner secrets never inherit, persistent workspace cwd) and `docker` backend (`--network none`, memory/cpu caps, throwaway container per call); workspace file tools (read/write/list) with path-traversal protection; `shell_exec` terminal (approval-gated **and** default-off); `spawn_agent` sub-agents (depth-1, shared router = one budget); Telegram inline Approve/Deny buttons (pulled forward from Phase 7 — the approval channel that makes write tools usable from a phone).

- [x] Sandbox escape attempts in tests stay contained — *path-traversal blocked (read/write/list), env scrubbing verified against planted secrets (TELEGRAM_BOT_TOKEN/BOT_STATE_TOKEN never visible to child processes), timeout kill, output caps; docker backend runs `--network none`*
- [x] Agent analyzes a CSV dataset end-to-end and produces a report file — *verified live on Cohere: write_file → python_exec over the file → correct sum/product reported (12.4s); sub-agent delegation verified live (research micro-task, 36.7s)*

## Phase 5 — Browser automation ✅ (v0.9.0)

Playwright-driven browser tool following the browser-use accessibility-tree pattern: navigate/click/type/extract/screenshot, action budgets, domain allowlists, approval on submissions (the "final click"). Screenshots as observations for the model.

Shipped in v0.9.0: eight `browser_*` tools over one lazily-launched headless chromium per conversation (`BrowserManager`, keyed by chat so the page survives across a chat's per-message agent rebuilds); numbered-element snapshots from a DOM walker (`data-woyo-ref` tags → real Playwright locator clicks/fills); `browser_submit` is the only path to submit-classified elements — WRITES_EXTERNAL, so the Telegram inline Approve/Deny button fires before the irreversible click (`browser_click` refuses those refs; `browser_type` refuses password+Enter); screenshots ride as one-shot image observations (router swaps to the vision model, payload stripped after the call that saw it, workspace copy saved for `send_file`); SSRF guard shared with fetch_url + optional domain allowlist enforced before navigation, after every action, and on form targets; bot-wall detection (status + challenge signatures) returns a typed observation that names the search+fetch fallback and forbids browser retries; budgets: 40 actions/session, 15-min TTL, idle close, 3-context LRU cap, downloads/service-workers/popups blocked.

- [x] Multi-page research workflow completes with sources — *verified on real chromium (local E2E): navigate → follow link → extract (token found) → back → form search via type+Enter → screenshot saved; citations flow through data.url like fetch_url*
- [x] A form-submission flow pauses for approval before the irreversible action — *verified twice: loop-level (denied browser_submit never clicks, observation says so) and real chromium (Sign-up POST refused by browser_click with needs_approval; through browser_submit it lands on the server)*
- [x] Bot-wall / failure handling degrades to search+fetch gracefully — *verified on real chromium: challenge page → typed error naming web_search/fetch_url and forbidding browser retries*

## Phase 6 — API server + web UI

FastAPI + SSE streaming the existing event bus; React UI (task view, plan, tool activity, approvals, history); auth (single-user first, multi-user later); tasks fully manageable from the UI. The CLI remains a first-class client.

- [ ] Same task runs identically via CLI and UI
- [ ] Approval requests delivered and answerable in the UI
- [ ] No secrets reachable from the client

## Phase 7 — Integrations

GitHub first (dogfood: woyo works on woyo — issues, PRs, releases), then Gmail / Calendar / Drive / Telegram per OAuth with per-integration grants, scoped tokens, and revocation. Each integration = Tool classes + a permission profile.

- [x] **Telegram reach + awareness + follow-through (v0.8, pulled forward):** the agent knows it lives inside Telegram (environment block: @username, current chat_id, known chats, real tool list, honest CANNOT-list); `list_chats` / `get_chat_info` / `send_telegram_message` / `send_document` reach any chat the bot is actually in (allowlisted, cross-chat sends approval-gated); `schedule_task` / `list_scheduled_tasks` / `cancel_scheduled_task` + `/tasks` turn "I'll do it later" into database rows that run on time and report back. Verified live on Cohere (Arabic): reminder scheduled with id quoted → fired on time → `done`.
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
