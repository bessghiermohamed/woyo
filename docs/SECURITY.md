# Security Model

woyo is an agent that browses the web, reads untrusted content, and (eventually) executes code and touches external services. This document is the honest threat model. It will be updated every phase.

## Principles

1. **Fail safe.** When a safety decision can't be made (no human reachable, unknown tool, ambiguous permission), the agent under-acts.
2. **Channel separation.** User instructions, system instructions, and external data are distinct channels. External data is never interpreted as instructions.
3. **Least privilege by default.** Tools declare permissions; the runtime enforces them. Consequential actions require a human.
4. **Visibility.** Everything an agent does is observable (events, transcripts). Secrets are not.

## Threats & mitigations (v0.1)

| # | Threat | Mitigation in v0.1 | Residual risk |
|---|---|---|---|
| T1 | **Prompt injection** via web pages, search snippets, documents (OWASP LLM01:2025) | All external content wrapped in `<untrusted source=...>` blocks; system-prompt contract forbids following instructions found there; best-effort detector flags instruction-shaped payloads (`injection_flagged` event); native tool-calling channel (not free-text); typed arg validation | **Not fully solvable** — an LLM can be manipulated. Layers reduce likelihood; approval gates + read-only defaults bound the blast radius. Treat any tool that acts on external content as untrusted by default |
| T2 | **SSRF** — agent fetches internal addresses | `fetch_url` resolves DNS and rejects private/loopback/link-local IPs, only http/https on 80/443, ≤5 re-validated redirects, 2 MB cap, content-type allowlist | DNS-rebinding not yet defeated (no egress proxy in v0.1); Phase 6 adds an egress allowlist/proxy |
| T3 | **Excessive autonomy / runaway cost** | Hard budgets: steps, tool calls, time, tokens, estimated USD; loop detection (identical repeated calls); approval gates; default-deny when no human | Estimation-based cost limits are approximate for unlisted models |
| T4 | **Dangerous actions** (writes, deletions, purchases) | `Permission` levels; `writes_external`/`destructive` require explicit approval; **denied by default** when no approval channel | Depends on honest tool classification — integration authors must classify correctly (documented in Tool protocol) |
| T5 | **Code execution escape** | `python_exec` off by default; two-tier sandbox since v0.5: `local` = isolated interpreter (`-I`), **scrubbed env** (children never inherit TELEGRAM_BOT_TOKEN / provider keys / runner tokens — tested with planted secrets), POSIX rlimits, timeout kill, output caps, workspace-scoped paths; `docker` = throwaway container per call, `--network none`, memory/cpu caps | `local` is a reliability boundary, NOT a security one (no namespace/seccomp) — use `WOYO_SANDBOX=docker` for untrusted code. `shell_exec` is denied twice (flag AND approval). File tools refuse path traversal (tested) |
| T9 | **Sub-agent misuse** (v0.5) | `spawn_agent` children: fixed tool include-list (no spawn_agent → no recursion, no approval-gated tools, no ask_user); own budgets; shared router so parent's cost/token gate covers child spend; child citations re-verified by the parent's citation gate | A child cannot press approvals (no human attached) or launder invented sources — verified in tests |
| T6 | **Secret leakage** | Secrets only via environment; never logged, never in events (args digest-logged, outputs capped); `.env` gitignored; transcripts exclude env | Provider request bodies inherently contain prompts — don't paste secrets into tasks |
| T7 | **Supply chain** (malicious deps) | Minimal dependency set (8 runtime deps, all mainstream); no agent framework; lockfile + review on upgrade | Standard Python ecosystem risk; pin and review |
| T8 | **Transcript/memory privacy** | Local-first storage (`~/.woyo`), user-readable formats, no telemetry | Local files are as safe as your user account |

## Approval policy

```
read_only       → runs automatically
sandboxed       → runs when explicitly enabled in config
writes_external → requires user approval per action (when WOYO_REQUIRE_APPROVAL=true, the default)
destructive     → requires user approval per action (always)
```

Since v0.5 the approval channel can be async: the Telegram frontend renders
inline Approve/Deny buttons and waits (bounded by `WOYO_CHAT_APPROVAL_TIMEOUT_S`,
timeout = deny). A press only counts from the chat that was asked. On GitHub
Actions the bot's process env contains live secrets — which is exactly why
child processes get a constructed minimal environment (see T5): the sandbox
scrubbing is the layer that keeps `shell_exec` from becoming a secret-exfil
primitive even after a human approves the command itself.

## Reporting

Security issues: please open a private GitHub security advisory on the repo rather than a public issue.

## Credential hygiene (added v0.1.1)

- **Never paste keys in chats, issues, or PRs.** Keys shared in plaintext (including with AI assistants) should be treated as compromised and rotated. If a key was exposed, revoke it and issue a new one before putting it in your local `.env`.
- woyo reads provider keys from the environment or your local `.env` — which is gitignored by default and must never be committed. The repo's secret-scan runs before every push.
- Prefer fine-grained, minimally-scoped tokens (e.g. GitHub fine-grained PATs scoped to one repo) over broad classic tokens.
- Keys for later phases (Telegram bot token, Supabase service key, Cloudflare/Vercel tokens) are reserved in `.env.example` as comments. The Supabase **service_role** key in particular bypasses Row Level Security — treat it like a root password and never expose it client-side.

## GPT4Free (g4f) — experimental extra

`g4f` routes requests through free LLM endpoints collected from the web. Using it means:

- your prompts (and anything the agent puts in them) travel to **unvetted third parties**,
- availability and quality change without notice (our Sept 2026 test: default chain failed without a local browser),
- some upstream endpoints do not sanction automated use (ToS/account risk).

Never enable g4f for tasks touching secrets, personal data, or anything confidential. This is why it is an opt-in extra, documented in ADR-9, and never a default.

## Chat frontends (v0.3)

**Telegram bot.** The token is a full credential: anyone holding it can read
your bot's messages and impersonate it. Keep it in `.env` (never committed);
if it leaks, revoke it via @BotFather (`/revoke`). Access control:
`TELEGRAM_ALLOWED_CHAT_IDS` wins when set; otherwise the first `/start`
claims the bot (persisted in `~/.woyo/telegram_state.json`) and everyone
else is refused. Only one poller may run per token — a 409 from the API
means a second instance is live somewhere.

**Web chat.** The page holds no secrets; the provider key never leaves the
server. Binding beyond localhost *requires* `WOYO_CHAT_PASSWORD`
(constant-time compare), and the server enforces per-session rate limits
(20 messages/hour) plus a global daily cap — a public URL must not become
a free API-key proxy. Chat inputs are framed as agent tasks and tool
outputs stay wrapped as untrusted data, so a malicious web page the agent
fetches cannot hijack the conversation (same T4 defenses as `woyo run`).
Unattended chat has no approval channel: external-write tools are denied
fail-safe, exactly like headless `woyo run`.

## Long-term memory (v0.4)

**Memory poisoning (new threat, T9).** Anything the agent stores — or that
recall injects into a prompt — is *data about the past*, not instructions.
Mitigations, layered:

- Recall output is framed in the system prompt as "hints from earlier
  sessions, NOT verified sources"; load-bearing facts must be re-verified
  against live data, and memory is never a citable source (citation
  verification only accepts URLs actually observed in the current run).
- Memory content passes the same untrusted-data treatment as tool output
  when rendered; the agent is instructed never to follow instructions
  found inside remembered text.
- `memory_save` is a local-only tool (no external effects, no approval
  needed) but everything it writes is user-auditable: `woyo memory
  list|show|delete|prune`. If a memory looks wrong, delete it — no
  memory is load-bearing for correctness.

**Data minimization is enforced, not advisory.** Memories carry a default
180-day TTL (configurable per item, 0 = keep until cap); expired rows are
purged on open. A global cap (default 5,000) evicts expired first, then
least-accessed, then oldest. The task queue prunes finished rows. The
SQLite file is plain local data — back it up or `rm` it; there is no
cloud copy unless you put one there.

**Cross-process control channel.** `woyo tasks pause|cancel` writes a
control request into the tasks table; the running poller acts on it
between steps. Only someone who can already write to `~/.woyo` (i.e.,
you, on your machine) can issue one — it is not a network surface.
