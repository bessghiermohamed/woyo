"""TaskRunner: execute queued tasks with checkpointing + external control.

Wires one Agent.run to a TaskStore row:
- every bus event is appended to task_events (audit trail)
- every loop step checkpoints RunState (resume after restart)
- the tasks.control column is polled between steps so another process
  can pause/cancel a running task
- approval_requested flips the row to waiting_approval so
  `woyo tasks list` shows exactly where a task is stuck
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from woyo.agent import Agent
from woyo.agent.results import RunResult
from woyo.config import Settings
from woyo.events import EventBus
from woyo.memory.longterm import MemoryStore
from woyo.models.router import ModelRouter
from woyo.store.tasks import TaskStore
from woyo.tools.builtin import build_default_registry

AgentFactory = Callable[[Settings, EventBus], Agent]


def _default_agent_factory(settings: Settings, bus: EventBus) -> Agent:
    memory: MemoryStore | None = None
    if settings.memory_enabled:
        from woyo.memory.longterm import build_memory_from_settings

        memory = build_memory_from_settings(settings)
    registry = build_default_registry(settings, bus=bus, memory=memory)
    router = ModelRouter(settings, bus=bus)
    return Agent(settings, router, registry, bus=bus, memory=memory)


class TaskRunner:
    """Runs tasks from the store, surviving crashes via checkpoints."""

    def __init__(
        self,
        settings: Settings,
        store: TaskStore,
        *,
        agent_factory: AgentFactory | None = None,
        event_sink: Callable[[str], None] | None = None,  # live-view hook
    ):
        self.settings = settings
        self.store = store
        self._agent_factory = agent_factory or _default_agent_factory
        self._event_sink = event_sink

    # ------------------------------------------------------------------
    async def run_task(
        self, task_id: int, *, approval_cb=None, fresh: bool = False
    ) -> RunResult:
        task = self.store.get(task_id)
        if task is None:
            raise ValueError(f"no task with id {task_id}")
        if task.status == "running":
            raise ValueError(
                f"task {task_id} is marked running in another process "
                "(use --recover if it crashed)"
            )

        self.store.clear_control(task_id)
        self.store.mark_running(task_id)

        bus = EventBus()
        agent = self._agent_factory(self.settings, bus)
        seq = 0

        def on_event(event) -> None:
            nonlocal seq
            seq += 1
            try:
                self.store.append_event(task_id, seq, event.kind, event.to_dict())
            except Exception:  # noqa: BLE001 — audit must not kill the run
                pass
            if event.kind == "approval_requested":
                self.store.set_status(task_id, "waiting_approval")
            elif event.kind in ("approval_result",) :
                self.store.set_status(task_id, "running")
            elif event.kind == "run_finished":
                pass  # status set below from the outcome
            if self._event_sink:
                self._event_sink(str(event))

        bus.subscribe(on_event)

        def checkpoint(state_dict: dict[str, Any]) -> None:
            if self.settings.task_checkpoint:
                self.store.save_checkpoint(task_id, state_dict)
                self.store.heartbeat(task_id)

        def control_poll() -> str | None:
            try:
                return self.store.poll_control(task_id)
            except Exception:  # noqa: BLE001 — control must not kill the run
                return None

        resume = None if fresh else task.checkpoint
        try:
            result = await agent.run(
                task.prompt,
                approval_cb=approval_cb,
                resume_state=resume,
                checkpoint_cb=checkpoint,
                control_poll=control_poll,
            )
        except Exception as exc:  # noqa: BLE001 — crashes become failed rows
            self.store.set_status(task_id, "failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.store.clear_control(task_id)

        self.store.save_result(task_id, result.to_dict())
        if result.outcome == "paused":
            self.store.set_status(task_id, "paused")
        elif result.outcome == "cancelled":
            self.store.set_status(task_id, "cancelled")
        elif result.outcome == "completed":
            self.store.set_status(task_id, "completed")
        else:
            self.store.set_status(
                task_id,
                "failed" if result.outcome == "failed" else "paused",
                error=result.outcome_detail or result.outcome,
            )
        return result

    # ------------------------------------------------------------------
    async def run_pending(self, *, limit: int = 10, approval_cb=None) -> list[RunResult]:
        """Recover stale rows, then run pending tasks FIFO."""
        self.store.recover_stale(self.settings.task_stale_minutes)
        results = []
        for task in self.store.list(status="pending", limit=limit * 4):
            if len(results) >= limit:
                break
            results.append(await self.run_task(task.id, approval_cb=approval_cb))
        return results
