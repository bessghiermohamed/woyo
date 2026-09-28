"""woyo CLI: run tasks, chat, inspect tools, verify setup.

The live view prints the event stream as it happens — the same stream the
future web UI will consume (ADR-8): plan, tool activity, approvals, limits.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from woyo import __version__
from woyo.config import PROVIDER_PRESETS, Settings
from woyo.errors import AgentError
from woyo.events import Event, EventBus
from woyo.memory.session import SessionStore
from woyo.models.router import ModelRouter
from woyo.tools.base import ToolRegistry, UserInteraction
from woyo.tools.builtin import build_default_registry

app = typer.Typer(
    name="woyo",
    help="A general-purpose AI agent you own.",
    no_args_is_help=True,
    add_completion=False,
)
memory_app = typer.Typer(help="Long-term memory: list, search, add, delete.", no_args_is_help=True)
tasks_app = typer.Typer(help="Persistent task queue: run, resume, cancel.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
app.add_typer(tasks_app, name="tasks")
console = Console()


class RichInteraction(UserInteraction):
    def ask(self, question: str, *, default: str | None = None) -> str:
        return Prompt.ask(f"[cyan]{question}[/cyan]", default=default or "")

    def confirm(self, question: str) -> bool:
        return Confirm.ask(f"[yellow]{question}[/yellow]")


class LivePrinter:
    """Prints one compact line per event — simple, robust with prompts."""

    def __init__(self) -> None:
        self.steps = 0
        self.tool_calls = 0

    def handle(self, event: Event) -> None:
        d = event.data
        if event.kind == "run_started":
            console.print(f"[dim]▶ task: {str(d.get('task'))[:120]}[/dim]")
        elif event.kind == "plan_created":
            console.print("[bold]plan:[/bold]")
            for s in d.get("steps", [])[:8]:
                console.print(f"  [dim]•[/dim] {s}")
        elif event.kind == "plan_fallback":
            console.print("[yellow]planner output unusable — fallback plan[/yellow]")
        elif event.kind == "llm_call":
            self.steps += 1
            console.print(f"[dim]· thinking (step {self.steps}, {d.get('model')})[/dim]")
        elif event.kind == "tool_call":
            console.print(f"[cyan]→ {d.get('tool')}[/cyan] [dim]{str(d.get('args'))[:140]}[/dim]")
        elif event.kind == "tool_result":
            self.tool_calls += 1
            mark = "[green]ok[/green]" if d.get("ok") else "[red]error[/red]"
            console.print(f"  [dim]{mark} · {d.get('chars', 0)} chars[/dim]")
        elif event.kind == "injection_flagged":
            console.print(
                f"[yellow]⚠ injection-shaped content flagged in {d.get('tool')} "
                "(treated as data)[/yellow]"
            )
        elif event.kind == "approval_requested":
            console.print(f"[yellow]? approval needed: {d.get('tool')}[/yellow]")
        elif event.kind == "limit_hit":
            console.print(f"[yellow]⏹ {d.get('limit')}[/yellow]")
        elif event.kind == "run_finished":
            pass  # the run report is printed by the caller


def _build_agent(settings: Settings, interaction: RichInteraction | None):
    from woyo.memory.longterm import build_memory_from_settings

    bus = EventBus()
    memory = build_memory_from_settings(settings)
    registry: ToolRegistry = build_default_registry(
        settings, bus=bus, interaction=interaction, memory=memory
    )
    router = ModelRouter(settings, bus=bus)
    from woyo.agent import Agent

    agent = Agent(settings, router, registry, bus=bus, memory=memory)
    return agent, bus


def _approval_prompt(tool_name: str, args_json: str) -> bool:
    console.print(
        Panel(
            f"[bold]{tool_name}[/bold]\n[args]{args_json[:600]}[/args]",
            title="Approval required — this tool acts externally",
            border_style="yellow",
        )
    )
    return Confirm.ask("Allow this action?", default=False)


async def _run_task(task: str, settings: Settings, auto_approve: bool = False):
    interaction = RichInteraction()
    agent, bus = _build_agent(settings, interaction)
    printer = LivePrinter()
    bus.subscribe(printer.handle)
    store = SessionStore(settings.sessions_path())
    session_id = store.new_session_id()

    def approval(tool_name: str, args_json: str) -> bool:
        if auto_approve:
            console.print(f"[yellow]auto-approved:[/yellow] {tool_name}")
            return True
        return _approval_prompt(tool_name, args_json)

    result = await agent.run(task, approval_cb=approval)
    store.append_events(session_id, result.events_log)
    return result, session_id


@app.command()
def run(
    task: str = typer.Argument(..., help="The goal for the agent"),
    max_steps: int = typer.Option(None, help="Override WOYO_MAX_STEPS"),
    yes: bool = typer.Option(False, "--yes", help="Auto-approve external actions"),
):
    """Run a single task and print the result."""
    settings = Settings()
    if max_steps:
        settings.max_steps = max_steps
    result, session_id = asyncio.run(_run_task(task, settings, auto_approve=yes))
    console.print()
    console.print(Panel(Markdown(result.final_answer), title="Result", border_style="green"))
    console.print(Panel(Text(result.summary()), title="Run report", border_style="dim"))
    console.print(f"[dim]session: {session_id}[/dim]")


@app.command()
def chat():
    """Interactive session: each message is a task; runs share one usage meter."""
    settings = Settings()
    interaction = RichInteraction()
    agent, bus = _build_agent(settings, interaction)
    printer = LivePrinter()
    bus.subscribe(printer.handle)
    store = SessionStore(settings.sessions_path())
    session_id = store.new_session_id()
    console.print(
        Panel(
            f"provider [bold]{settings.provider}[/bold] · model [bold]{settings.model}[/bold]\n"
            "Commands: /usage · /tools · /new · /exit",
            title=f"woyo chat — {session_id}",
            border_style="cyan",
        )
    )
    while True:
        try:
            user_input = Prompt.ask("[bold green]you ›[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/new":
            agent, bus = _build_agent(settings, interaction)
            bus.subscribe(printer.handle)
            session_id = store.new_session_id()
            console.print(f"[dim]new session: {session_id}[/dim]")
            continue
        if user_input == "/usage":
            console.print_json(json.dumps(agent.router.usage_summary()))
            continue
        if user_input == "/tools":
            _print_tools(agent.registry)
            continue
        try:
            result = asyncio.run(agent.run(user_input, approval_cb=_approval_prompt))
        except Exception as exc:  # noqa: BLE001 — CLI must not die
            console.print(f"[red]error:[/red] {exc}")
            continue
        store.append_events(session_id, result.events_log)
        console.print(Panel(Markdown(result.final_answer), title="woyo", border_style="cyan"))
        console.print(
            f"[dim]{result.outcome} · {result.steps} steps · "
            f"${result.usage.cost_usd_est:.4f}[/dim]"
        )


@app.command()
def telegram():
    """Run the Telegram chat bot (long-polling; Ctrl-C to stop)."""
    from woyo.chat.telegram import TelegramBot, telegram_credentials

    settings = Settings()
    token, allowed = telegram_credentials()
    if not token:
        console.print(
            Panel(
                "TELEGRAM_BOT_TOKEN not found.\n\n"
                "1. Talk to @BotFather on Telegram → /newbot → copy the token\n"
                "2. Put it in your .env:  TELEGRAM_BOT_TOKEN=123456:ABC-DEF...\n"
                "3. Optional: TELEGRAM_ALLOWED_CHAT_IDS=123456789",
                title="woyo telegram", border_style="yellow",
            )
        )
        raise typer.Exit(1)
    bot = TelegramBot(settings, token, allowed)
    try:
        asyncio.run(bot.run_forever())
    except KeyboardInterrupt:
        console.print("[dim]bot stopped[/dim]")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Bind address (0.0.0.0 needs a passcode)"),
    port: int = typer.Option(7860, help="Port (7860 is the HF Spaces default)"),
):
    """Serve the web chat UI (phone-friendly, passcode-gated when public)."""
    from woyo.chat.web import serve

    try:
        serve(Settings(), host=host, port=port)
    except AgentError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _print_tools(registry: ToolRegistry) -> None:
    table = Table(title="Tools")
    table.add_column("name")
    table.add_column("permission")
    table.add_column("description")
    for item in registry.safety_overview():
        table.add_row(item["name"], item["permission"], item["description"])
    console.print(table)


@app.command()
def tools():
    """List available tools and their permission levels."""
    settings = Settings()
    registry = build_default_registry(settings)
    _print_tools(registry)


@app.command()
def doctor():
    """Diagnose your woyo setup (provider, keys, search, optional deps)."""
    settings = Settings()
    checks: list[tuple[str, bool, str]] = []

    key = settings.resolved_api_key()
    provider = settings.provider
    if provider == "mock":
        checks.append(("provider", True, "mock (no network)"))
    elif provider == "g4f":
        checks.append(("provider", True, "g4f — no key needed (experimental)"))
    elif key:
        checks.append(("provider", True, f"{provider} — key found"))
    else:
        preset = PROVIDER_PRESETS.get(provider, {})
        env_name = preset.get("key_env") or "WOYO_API_KEY"
        checks.append(("provider", False, f"{provider} — set {env_name} (see .env.example)"))

    from woyo.config import env_value

    if env_value("TAVILY_API_KEY"):
        checks.append(("search", True, "tavily (free credits monthly)"))
    elif env_value("BRAVE_API_KEY"):
        checks.append(("search", True, "brave"))
    else:
        try:
            import ddgs  # noqa: F401

            checks.append(("search", True, "duckduckgo (keyless, via ddgs)"))
        except ImportError:
            checks.append(
                ("search", False, "no backend — set TAVILY_API_KEY/BRAVE_API_KEY "
                 "or pip install 'woyo[search]'")
            )

    try:
        import trafilatura  # noqa: F401

        checks.append(("extraction", True, "trafilatura"))
    except ImportError:
        checks.append(("extraction", False, "optional: pip install 'woyo[extract]'"))

    try:
        import anthropic  # noqa: F401

        checks.append(("anthropic", True, "available"))
    except ImportError:
        checks.append(("anthropic", False, "optional: pip install 'woyo[anthropic]'"))

    try:
        import g4f  # noqa: F401

        checks.append(
            ("g4f", True, "available (experimental — see docs/SECURITY.md)")
        )
    except ImportError:
        checks.append(("g4f", False, "optional: pip install 'woyo[g4f]' (experimental)"))

    try:
        import simpleeval  # noqa: F401

        checks.append(("calculate", True, "simpleeval"))
    except ImportError:
        checks.append(("calculate", False, "missing — pip install -e ."))

    from woyo.chat.telegram import telegram_credentials

    tg_token, tg_allowed = telegram_credentials()
    if tg_token:
        scope = f"allowed ids: {sorted(tg_allowed)}" if tg_allowed else "first /start claims the bot"
        checks.append(("telegram", True, f"token found ({scope})"))
    else:
        checks.append(("telegram", False, "no token — set TELEGRAM_BOT_TOKEN (optional)"))

    if settings.chat_password:
        checks.append(("web chat", True, "passcode set — safe to expose publicly"))
    else:
        checks.append(("web chat", True, "no passcode — localhost only (WOYO_CHAT_PASSWORD)"))

    if settings.cache_enabled:
        from woyo.tools.http_cache import build_cache_from_settings

        try:
            cache = build_cache_from_settings(settings)
            stats = cache.stats() if cache else {}
            cache.close() if cache else None
            checks.append(
                (
                    "cache",
                    True,
                    f"enabled — {stats.get('entries', 0)} entries, "
                    f"{stats.get('bytes', 0) // 1024} KB "
                    f"({settings.cache_dir}/cache.sqlite3)",
                )
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics must not crash
            checks.append(("cache", False, f"enabled but unusable: {exc}"))
    else:
        checks.append(("cache", False, "disabled (WOYO_CACHE_ENABLED=false)"))

    if settings.memory_enabled:
        try:
            from woyo.memory.longterm import build_memory_from_settings

            memory = build_memory_from_settings(settings)
            stats = memory.stats() if memory else {}
            memory.close() if memory else None
            checks.append(
                (
                    "memory",
                    True,
                    f"{stats.get('total', 0)} items, embedder {stats.get('embedder')}, "
                    f"recall top-{settings.memory_recall_k} "
                    f"(cap {settings.memory_max_items})",
                )
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics must not crash
            checks.append(("memory", False, f"enabled but unusable: {exc}"))
    else:
        checks.append(("memory", False, "disabled (WOYO_MEMORY_ENABLED=false)"))

    try:
        from woyo.store.db import db_path_from_settings
        from woyo.store.tasks import TaskStore

        task_store = TaskStore(db_path_from_settings(settings))
        counts: dict[str, int] = {}
        for t in task_store.list(limit=1000):
            counts[t.status] = counts.get(t.status, 0) + 1
        task_store.close()
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "empty"
        checks.append(("tasks", True, f"{summary} ({settings.db_path})"))
    except Exception as exc:  # noqa: BLE001 — diagnostics must not crash
        checks.append(("tasks", False, f"unusable: {exc}"))

    checks.append(
        (
            "code exec",
            settings.enable_code_exec,
            "enabled (DEV-GRADE sandbox, see docs/SECURITY.md)"
            if settings.enable_code_exec
            else "disabled (default, safe)",
        )
    )

    table = Table(title=f"woyo doctor — v{__version__}")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]ok[/green]" if ok else "[yellow]!![/yellow]", escape(detail))
    console.print(table)


# =====================================================================
# woyo memory — long-term memory management (Phase 3)
# =====================================================================

def _memory_store():
    from woyo.memory.longterm import MemoryStore, build_embedder
    from woyo.store.db import db_path_from_settings

    settings = Settings()
    if not settings.memory_enabled:
        console.print("[yellow]memory is disabled (WOYO_MEMORY_ENABLED=false)[/yellow]")
        raise typer.Exit(1)
    return MemoryStore(
        db_path_from_settings(settings),
        embedder=build_embedder(settings),
        max_items=settings.memory_max_items,
        default_ttl_days=settings.memory_default_ttl_days,
        min_similarity=settings.memory_min_similarity,
    )


@memory_app.command("list")
def memory_list(
    kind: str = typer.Option(None, help="Filter: fact | preference | note"),
    limit: int = typer.Option(20, help="Rows to show"),
    all_items: bool = typer.Option(False, "--all", help="Include expired"),
):
    """List stored memories (newest first)."""
    store = _memory_store()
    rows = store.list(kind=kind, limit=limit, include_expired=all_items)
    if not rows:
        console.print("[dim]no memories stored yet[/dim]")
        return
    table = Table(title=f"woyo memory — {len(rows)} shown")
    table.add_column("id", justify="right")
    table.add_column("kind")
    table.add_column("content")
    table.add_column("accesses", justify="right")
    table.add_column("expires")
    for r in rows:
        content = r["content"] if len(r["content"]) <= 90 else r["content"][:87] + "..."
        table.add_row(
            str(r["id"]), r["kind"], escape(content), str(r["access_count"]),
            r["expires_at"] or "never",
        )
    console.print(table)


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="The fact/note to remember"),
    kind: str = typer.Option("note", help="fact | preference | note"),
    ttl_days: int = typer.Option(None, help="Days until expiry (default from settings)"),
):
    """Store a memory directly."""
    store = _memory_store()
    memory_id = asyncio.run(store.remember(content, kind=kind, source="cli",
                                            ttl_days=ttl_days))
    console.print(f"[green]saved[/green] memory id={memory_id} (kind={kind})")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to look for"),
    k: int = typer.Option(5, help="Maximum hits"),
):
    """Semantic search over stored memories."""
    store = _memory_store()
    hits = asyncio.run(store.recall(query, k=k))
    if not hits:
        console.print("[dim]no relevant memories[/dim]")
        return
    for h in hits:
        console.print(f"[cyan]#{h.id}[/cyan] ({h.kind}, sim {h.similarity:.2f}) {h.content}")


@memory_app.command("show")
def memory_show(memory_id: int = typer.Argument(..., help="Memory id")):
    """Show one memory in full."""
    store = _memory_store()
    row = store.get(memory_id)
    if not row:
        console.print(f"[red]no memory with id={memory_id}[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(row, default=str))


@memory_app.command("delete")
def memory_delete(
    memory_id: int = typer.Argument(..., help="Memory id"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
):
    """Delete a memory (your data, your call)."""
    store = _memory_store()
    row = store.get(memory_id)
    if not row:
        console.print(f"[red]no memory with id={memory_id}[/red]")
        raise typer.Exit(1)
    if not yes and not Confirm.ask(f"Delete memory #{memory_id}: {row['content'][:80]}?", default=False):
        return
    if store.delete(memory_id):
        console.print(f"[green]deleted[/green] memory #{memory_id}")


@memory_app.command("prune")
def memory_prune(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would go"),
):
    """Delete expired memories (data minimization)."""
    store = _memory_store()
    if dry_run:
        rows = store.list(limit=10_000, include_expired=True)
        n = sum(1 for r in rows if r["expires_at"] and r["expires_at"] <= _utc_now())
        console.print(f"[dim]{n} expired memor{'y' if n == 1 else 'ies'} would be deleted[/dim]")
        return
    n = store.purge_expired()
    console.print(f"[green]pruned[/green] {n} expired memor{'y' if n == 1 else 'ies'}")


@memory_app.command("stats")
def memory_stats():
    """Memory size, kinds, embedder, cap."""
    store = _memory_store()
    console.print_json(json.dumps(store.stats(), default=str))


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


# =====================================================================
# woyo tasks — persistent task queue (Phase 3)
# =====================================================================

def _task_store():
    from woyo.store.db import db_path_from_settings
    from woyo.store.tasks import TaskStore

    return TaskStore(db_path_from_settings(Settings()))


_STATUS_COLORS = {
    "pending": "white", "running": "cyan", "waiting_approval": "yellow",
    "paused": "magenta", "completed": "green", "failed": "red", "cancelled": "dim",
}


@tasks_app.command("list")
def tasks_list(
    status: str = typer.Option(None, help="Filter by status"),
    limit: int = typer.Option(20, help="Rows to show"),
):
    """List tasks (newest first)."""
    store = _task_store()
    if status:
        store.recover_stale(Settings().task_stale_minutes)
    rows = store.list(status=status, limit=limit)
    if not rows:
        console.print("[dim]no tasks[/dim]")
        return
    table = Table(title=f"woyo tasks — {len(rows)} shown")
    table.add_column("id", justify="right")
    table.add_column("status")
    table.add_column("attempts", justify="right")
    table.add_column("title")
    table.add_column("updated")
    for t in rows:
        color = _STATUS_COLORS.get(t.status, "white")
        table.add_row(
            str(t.id), f"[{color}]{t.status}[/{color}]", str(t.attempts),
            escape(t.title[:70]), t.updated_at[:16],
        )
    console.print(table)


@tasks_app.command("add")
def tasks_add(
    task: str = typer.Argument(..., help="The goal for the agent"),
    run_now: bool = typer.Option(False, "--run", help="Run immediately after queuing"),
):
    """Queue a task (picked up by `woyo tasks run`)."""
    store = _task_store()
    t = store.create(task, source="cli")
    console.print(f"[green]queued[/green] task #{t.id}: {t.title[:80]}")
    if run_now:
        _run_one(t.id, auto_approve=False)


@tasks_app.command("run")
def tasks_run(
    task_id: int = typer.Option(None, "--id", help="Run/resume a specific task"),
    all_pending: bool = typer.Option(False, "--all", help="Run every pending task"),
    limit: int = typer.Option(5, help="Max pending tasks per invocation"),
    recover: bool = typer.Option(False, "--recover", help="Recover a crashed running row"),
    yes: bool = typer.Option(False, "--yes", help="Auto-approve external actions"),
):
    """Run pending tasks (or resume a paused/interrupted one)."""
    if task_id is not None:
        if recover:
            _task_store().set_status(task_id, "paused", error="recovered by user")
        _run_one(task_id, auto_approve=yes)
        return
    store = _task_store()
    recovered = store.recover_stale(Settings().task_stale_minutes)
    for task_id in recovered:
        console.print(f"[magenta]recovered interrupted task #{task_id} (paused)[/magenta]")
    if not all_pending:
        pending = store.list(status="pending", limit=1)
        if not pending:
            console.print("[dim]no pending tasks — `woyo tasks add \"...\"` to queue one[/dim]")
            return
        _run_one(pending[0].id, auto_approve=yes)
        return
    from woyo.agent.runner import TaskRunner

    runner = TaskRunner(Settings(), store, event_sink=_live_line)
    results = asyncio.run(runner.run_pending(limit=limit, approval_cb=_auto_or_prompt(yes)))
    console.print(f"[dim]{len(results)} task(s) processed[/dim]")


def _auto_or_prompt(auto: bool):
    if auto:
        return lambda tool_name, args_json: (
            console.print(f"[yellow]auto-approved:[/yellow] {tool_name}") or True
        )
    return _approval_prompt


def _live_line(line: str) -> None:
    console.print(Text(line[:160], style="dim"))


def _run_one(task_id: int, *, auto_approve: bool) -> None:
    from woyo.agent.runner import TaskRunner

    store = _task_store()
    runner = TaskRunner(Settings(), store, event_sink=_live_line)
    try:
        result = asyncio.run(
            runner.run_task(task_id, approval_cb=_auto_or_prompt(auto_approve))
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print()
    console.print(Panel(Markdown(result.final_answer), title=f"task #{task_id}", border_style="green"))
    console.print(Panel(Text(result.summary()), title="Run report", border_style="dim"))


@tasks_app.command("show")
def tasks_show(
    task_id: int = typer.Argument(..., help="Task id"),
    events: bool = typer.Option(False, "--events", help="Include the event log"),
):
    """Inspect a task: status, result, checkpoint state."""
    store = _task_store()
    t = store.get(task_id)
    if not t:
        console.print(f"[red]no task with id={task_id}[/red]")
        raise typer.Exit(1)
    body = {
        "id": t.id, "status": t.status, "attempts": t.attempts,
        "created": t.created_at, "updated": t.updated_at,
        "finished": t.finished_at, "error": t.error,
        "prompt": t.prompt, "result": t.result,
        "checkpoint_steps": (t.checkpoint or {}).get("steps"),
    }
    console.print_json(json.dumps(body, default=str))
    if events:
        for e in store.events(task_id, limit=50):
            console.print(f"[dim]#{e['seq']}[/dim] {e['kind']} {escape(str(e['data'])[:120])}")


@tasks_app.command("pause")
def tasks_pause(task_id: int = typer.Argument(..., help="Task id")):
    """Ask a running task to pause (checkpointed, resumable)."""
    _task_store().request_control(task_id, "pause")
    console.print(f"[magenta]pause requested[/magenta] for task #{task_id}")


@tasks_app.command("cancel")
def tasks_cancel(task_id: int = typer.Argument(..., help="Task id")):
    """Ask a running task to cancel."""
    _task_store().request_control(task_id, "cancel")
    console.print(f"[yellow]cancel requested[/yellow] for task #{task_id}")


@tasks_app.command("retry")
def tasks_retry(task_id: int = typer.Argument(..., help="Task id")):
    """Re-queue a finished/failed/cancelled task from scratch."""
    t = _task_store().retry(task_id)
    if t is None:
        console.print(f"[red]no task with id={task_id}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]re-queued[/green] task #{t.id} (attempt {t.attempts + 1})")


@tasks_app.command("prune")
def tasks_prune(
    days: int = typer.Option(30, help="Delete finished tasks older than this"),
    keep_failed: bool = typer.Option(True, help="Keep failed rows for debugging"),
):
    """Delete old finished tasks."""
    n = _task_store().prune(days=days, keep_failed=keep_failed)
    console.print(f"[green]pruned[/green] {n} finished task(s)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
