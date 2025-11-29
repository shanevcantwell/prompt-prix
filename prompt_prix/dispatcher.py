"""
WorkStealingDispatcher - reusable parallel execution across multiple servers.

Pattern extracted from handlers.py:send_single_prompt for reuse.
Efficiently distributes work items to available servers.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Coroutine, Optional, Protocol, TypeVar

from prompt_prix.core import ServerPool

logger = logging.getLogger(__name__)


# Type for optional cancellation check function
CancellationCheck = Optional[Callable[[], bool]]


class WorkItem(Protocol):
    """Protocol for dispatchable work items."""

    @property
    def model_id(self) -> str:
        """Model ID this work item should run on."""
        ...


T = TypeVar("T", bound=WorkItem)


class WorkStealingDispatcher:
    """
    Reusable work-stealing dispatcher.

    Distributes work items across available servers, keeping all servers
    busy and matching work to servers that have the required models.

    Design per CLAUDE.md:
    - Explicit state management (tracks active tasks, completed count)
    - Observable (yields for UI updates)
    - Fail loudly (exceptions propagate from execute_fn)
    """

    def __init__(self, pool: ServerPool):
        """
        Initialize dispatcher with server pool.

        Args:
            pool: ServerPool managing available servers
        """
        self.pool = pool

    async def dispatch(
        self,
        work_items: list[T],
        execute_fn: Callable[[T, str], Coroutine[None, None, None]],
        should_cancel: CancellationCheck = None,
    ) -> AsyncGenerator[int, None]:
        """
        Execute work items across available servers.

        Work-stealing algorithm:
        1. Queue all work items
        2. For each idle server, find work it can run
        3. Spawn async task for matched work
        4. Yield periodically for UI updates
        5. Clean up completed tasks

        Args:
            work_items: List of items to process (must have model_id property)
            execute_fn: async fn(item, server_url) to execute each item
            should_cancel: Optional fn() -> bool to check for cancellation

        Yields:
            int: Number of completed items (for progress tracking)
        """
        work_queue = list(work_items)
        active_tasks: dict[str, asyncio.Task] = {}  # server_url -> task
        completed = 0
        total = len(work_items)

        # Initial yield
        yield completed

        while work_queue or active_tasks:
            # Check for cancellation - stop assigning new work
            if should_cancel and should_cancel():
                logger.info("Cancellation requested, clearing work queue")
                work_queue.clear()
            # Assign work to idle servers
            for server_url, server in self.pool.servers.items():
                if server.is_busy:
                    continue

                # Find first work item this server can run
                matched_item = None
                for item in work_queue:
                    if item.model_id in server.available_models:
                        matched_item = item
                        break

                if matched_item:
                    work_queue.remove(matched_item)
                    await self.pool.acquire_server(server_url)
                    task = asyncio.create_task(
                        self._run_and_release(matched_item, server_url, execute_fn)
                    )
                    active_tasks[server_url] = task

            # Wait a bit, yield for UI update
            await asyncio.sleep(0.1)
            yield completed

            # Clean up completed tasks
            for url in list(active_tasks.keys()):
                if active_tasks[url].done():
                    # Check for exceptions
                    try:
                        active_tasks[url].result()
                    except Exception:
                        pass  # Error handling in execute_fn
                    completed += 1
                    del active_tasks[url]

            # Refresh manifests if stuck (work but no servers can run it)
            if work_queue and not active_tasks:
                await self.pool.refresh_all_manifests()

        # Final yield with all completed
        yield completed

    async def _run_and_release(
        self,
        item: T,
        server_url: str,
        execute_fn: Callable[[T, str], Coroutine[None, None, None]],
    ) -> None:
        """
        Execute work item and release server when done.

        Ensures server is released even if execute_fn raises.
        """
        logger.debug(f"Acquired server {server_url} for model {item.model_id}")
        try:
            await execute_fn(item, server_url)
        except Exception as e:
            logger.warning(f"Task failed on {server_url}: {e}")
            raise
        finally:
            logger.debug(f"Releasing server {server_url}")
            self.pool.release_server(server_url)


@dataclass
class SimpleWorkItem:
    """Simple work item for basic use cases."""

    model_id: str
