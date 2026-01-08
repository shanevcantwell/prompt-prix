"""
Batch-based scheduler for model evaluation.

Executes tests in model batches - all tests for one model run together
on one server before moving to the next model. This minimizes VRAM
swapping and ensures fair comparison conditions.

Design principles:
- ModelBatch is the atomic unit, not (test, model) pairs
- Server affinity: prefer servers where model is already loaded
- Explicit state: manifest vs loaded is not conflated
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """
    State of a single LM Studio server.

    Separates what CAN run (manifest) from what IS running (loaded).
    LM Studio supports loading multiple models into VRAM simultaneously.
    """
    url: str
    manifest_models: list[str] = field(default_factory=list)  # From /v1/models
    loaded_models: list[str] = field(default_factory=list)  # From /api/v0/models (currently in VRAM)
    is_busy: bool = False


@dataclass
class ModelBatch:
    """
    A batch of tests to run against a single model.

    The batch is the atomic scheduling unit - all tests run together
    on one server before the server is released.
    """
    model_id: str
    test_ids: list[str]
    assigned_server: Optional[str] = None


@dataclass
class BatchProgress:
    """Progress update from batch runner."""
    completed_tests: int
    total_tests: int
    completed_batches: int
    total_batches: int
    active_models: list[str]  # Models currently running


class ServerPool:
    """
    Manages multiple LM Studio servers with explicit load state tracking.

    Queries both the OpenAI-compatible manifest endpoint and LM Studio's
    native API for load state.
    """

    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerState] = {
            url: ServerState(url=url) for url in server_urls
        }
        self._locks: dict[str, asyncio.Lock] = {
            url: asyncio.Lock() for url in server_urls
        }

    async def refresh(self) -> None:
        """Refresh both manifest and load state for all servers."""
        tasks = [self._refresh_server(url) for url in self.servers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_server(self, server_url: str) -> None:
        """Refresh manifest and load state for a single server."""
        import httpx

        server = self.servers[server_url]

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get manifest (OpenAI-compatible endpoint)
            try:
                resp = await client.get(f"{server_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    server.manifest_models = [m["id"] for m in data.get("data", [])]
            except Exception as e:
                logger.debug(f"Failed to get manifest from {server_url}: {e}")
                server.manifest_models = []

            # Get load state (LM Studio native API)
            try:
                resp = await client.get(f"{server_url}/api/v0/models")
                if resp.status_code == 200:
                    data = resp.json()
                    # Collect ALL loaded models (LM Studio can load multiple into VRAM)
                    loaded = [
                        model.get("id")
                        for model in data.get("data", [])
                        if model.get("state") == "loaded" and model.get("id")
                    ]
                    server.loaded_models = loaded
            except Exception as e:
                logger.debug(f"Failed to get load state from {server_url}: {e}")
                server.loaded_models = []

    def find_server(
        self,
        model_id: str,
        require_loaded: bool = False,
        preferred_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Find best server for a model.

        Priority:
        1. Preferred URL if specified and model available there
        2. Idle server with model already loaded (no swap needed)
        3. Idle server with model in manifest (will JIT load)

        Args:
            model_id: Model to find server for
            require_loaded: If True, only return servers where model is loaded
            preferred_url: If set, force this server (for GPU prefix routing)

        Returns:
            Server URL or None if no suitable server available
        """
        # If preferred URL specified, use it if model is available there
        if preferred_url and preferred_url in self.servers:
            server = self.servers[preferred_url]
            if not server.is_busy:
                if model_id in server.loaded_models or model_id in server.manifest_models:
                    return preferred_url

        # First pass: prefer server where model is already loaded
        for url, server in self.servers.items():
            if server.is_busy:
                continue
            if model_id in server.loaded_models:
                return url

        # Second pass: any server with model in manifest (unless require_loaded)
        if not require_loaded:
            for url, server in self.servers.items():
                if server.is_busy:
                    continue
                if model_id in server.manifest_models:
                    return url

        return None

    def get_available_models(self, only_loaded: bool = False) -> set[str]:
        """
        Get all models that can be run.

        Args:
            only_loaded: If True, only return models currently loaded

        Returns:
            Set of model IDs
        """
        if only_loaded:
            result = set()
            for server in self.servers.values():
                result.update(server.loaded_models)
            return result
        else:
            result = set()
            for server in self.servers.values():
                result.update(server.manifest_models)
            return result

    async def acquire(self, server_url: str) -> None:
        """Mark server as busy."""
        await self._locks[server_url].acquire()
        self.servers[server_url].is_busy = True

    def release(self, server_url: str) -> None:
        """Mark server as available."""
        self.servers[server_url].is_busy = False
        try:
            self._locks[server_url].release()
        except RuntimeError:
            pass  # Lock wasn't held


# Type alias for the test execution function
# (test_id, model_id, server_url) -> None
TestExecutor = Callable[[str, str, str], Coroutine[None, None, None]]


class BatchRunner:
    """
    Execute model batches across multiple servers.

    Each model's tests run as a batch on one server. This ensures:
    - Fair comparison: all tests for a model run under same conditions
    - Minimal swapping: model loaded once, all tests run, then released
    - Multi-GPU utilization: different models run in parallel on different GPUs
    """

    def __init__(self, pool: ServerPool):
        self.pool = pool
        self._should_stop = False

    def request_stop(self) -> None:
        """Request graceful stop after current batch completes."""
        self._should_stop = True

    def clear_stop(self) -> None:
        """Clear stop flag for new run."""
        self._should_stop = False

    async def run(
        self,
        models: list[str],
        test_ids: list[str],
        execute_fn: TestExecutor,
    ) -> AsyncGenerator[BatchProgress, None]:
        """
        Run all tests against all models, batched by model.

        Args:
            models: List of model IDs to test
            test_ids: List of test IDs to run against each model
            execute_fn: async fn(test_id, model_id, server_url) to execute each test

        Yields:
            BatchProgress updates for UI
        """
        self.clear_stop()

        # Create batches - one per model
        batches = [ModelBatch(model_id=m, test_ids=list(test_ids)) for m in models]
        pending = list(batches)
        active: dict[str, tuple[asyncio.Task, ModelBatch]] = {}  # server_url -> (task, batch)

        completed_tests = 0
        completed_batches = 0
        total_tests = len(models) * len(test_ids)
        total_batches = len(models)

        # Initial progress
        yield BatchProgress(
            completed_tests=0,
            total_tests=total_tests,
            completed_batches=0,
            total_batches=total_batches,
            active_models=[]
        )

        while pending or active:
            # Check for stop request
            if self._should_stop:
                logger.info("Stop requested, clearing pending batches")
                pending.clear()

            # Assign pending batches to idle servers
            for batch in list(pending):
                from prompt_prix import state  # Late import to avoid circular dependency
                hint = state.get_server_hint(batch.model_id)
                server_url = self.pool.find_server(batch.model_id, preferred_url=hint)
                if server_url:
                    pending.remove(batch)
                    batch.assigned_server = server_url
                    await self.pool.acquire(server_url)

                    task = asyncio.create_task(
                        self._run_batch(batch, server_url, execute_fn)
                    )
                    active[server_url] = (task, batch)
                    logger.info(f"Started batch for {batch.model_id} on {server_url}")

            # Yield progress
            yield BatchProgress(
                completed_tests=completed_tests,
                total_tests=total_tests,
                completed_batches=completed_batches,
                total_batches=total_batches,
                active_models=[b.model_id for _, b in active.values()]
            )

            # Wait briefly
            await asyncio.sleep(0.1)

            # Check for completed batches
            for url in list(active.keys()):
                task, batch = active[url]
                if task.done():
                    try:
                        tests_completed = task.result()
                        completed_tests += tests_completed
                        completed_batches += 1
                        logger.info(f"Completed batch for {batch.model_id}: {tests_completed} tests")
                    except Exception as e:
                        logger.error(f"Batch failed for {batch.model_id}: {e}")
                        completed_batches += 1  # Count as done even if failed

                    del active[url]
                    self.pool.release(url)

            # If we have pending work but no active tasks and no servers available,
            # refresh server state
            if pending and not active:
                await self.pool.refresh()

        # Final progress
        yield BatchProgress(
            completed_tests=completed_tests,
            total_tests=total_tests,
            completed_batches=completed_batches,
            total_batches=total_batches,
            active_models=[]
        )

    async def _run_batch(
        self,
        batch: ModelBatch,
        server_url: str,
        execute_fn: TestExecutor,
    ) -> int:
        """
        Run all tests in a batch sequentially on one server.

        Returns number of tests completed.
        """
        completed = 0
        for test_id in batch.test_ids:
            if self._should_stop:
                break
            try:
                await execute_fn(test_id, batch.model_id, server_url)
                completed += 1
            except Exception as e:
                logger.warning(f"Test {test_id} failed for {batch.model_id}: {e}")
                completed += 1  # Count as completed (with error status)
        return completed
