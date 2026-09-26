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
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from woyo import __version__
from woyo.config import PROVIDER_PRESETS, Settings
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
    bus = EventBus()
    registry: ToolRegistry = build_default_registry(
        settings, bus=bus, interaction=interaction
    )
    router = ModelRouter(settings, bus=bus)
    from woyo.agent import Agent

    agent = Agent(settings, router, registry, bus=bus)
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
    elif key:
        checks.append(("provider", True, f"{provider} — key found"))
    else:
        preset = PROVIDER_PRESETS.get(provider, {})
        env_name = preset.get("key_env") or "WOYO_API_KEY"
        checks.append(("provider", False, f"{provider} — set {env_name} (see .env.example)"))

    import os

    if os.environ.get("TAVILY_API_KEY"):
        checks.append(("search", True, "tavily (free credits monthly)"))
    elif os.environ.get("BRAVE_API_KEY"):
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
        import simpleeval  # noqa: F401

        checks.append(("calculate", True, "simpleeval"))
    except ImportError:
        checks.append(("calculate", False, "missing — pip install -e ."))

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
        table.add_row(name, "[green]ok[/green]" if ok else "[yellow]!![/yellow]", detail)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
