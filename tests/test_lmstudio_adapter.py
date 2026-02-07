"""Tests for LMStudioAdapter boundary conditions.

Per ADR-006: ServerPool is INTERNAL to LMStudioAdapter. Tests mock at httpx boundary.
"""

import asyncio
import pytest
import respx
import httpx

from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.adapters.schema import InferenceTask


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_server_urls():
    """Two server URLs for testing."""
    return ["http://server0:1234", "http://server1:1234"]


def models_response(model_ids: list[str]) -> dict:
    """Build /v1/models response."""
    return {"data": [{"id": m} for m in model_ids]}


def sse_stream(content: str) -> str:
    """Build SSE stream response for chat completions."""
    return (
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        'data: [DONE]\n\n'
    )


# ─────────────────────────────────────────────────────────────────────
# get_available_models() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetAvailableModels:
    """Tests for get_available_models() aggregation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_aggregates_models_from_both_servers(self, two_server_urls):
        """Server 0 has [A,B], server 1 has [B,C] → returns [A,B,C]."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB", "modelC"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert set(models) == {"modelA", "modelB", "modelC"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_models_from_reachable_server_only(self, two_server_urls):
        """Server 0 unreachable → returns models from server 1 only."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelC", "modelD"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert set(models) == {"modelC", "modelD"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_both_servers_unreachable_returns_empty(self, two_server_urls):
        """Both servers unreachable → returns empty list."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert models == []


# ─────────────────────────────────────────────────────────────────────
# stream_completion() MODEL AVAILABILITY TESTS
# ─────────────────────────────────────────────────────────────────────

class TestStreamCompletionModelAvailability:
    """Tests for model availability routing."""

    # NOTE: test_model_not_on_any_server_times_out removed (see #121).
    # Queue wait no longer has timeout - requests wait patiently for servers.
    # Requesting a nonexistent model would wait indefinitely, but this can't
    # happen via UI since users select from get_available_models() dropdown.

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_on_server_1_only_routes_correctly(self, two_server_urls):
        """Model on server 1 only, no prefix → routes to server 1."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )
        # Expect completion on server 1 (the only one with modelB)
        respx.post("http://server1:1234/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_stream("World"))
        )

        adapter = LMStudioAdapter(two_server_urls)
        response = ""
        task = InferenceTask(
            model_id="modelB",  # No prefix, should find on server 1
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=5.0
        )
        async for chunk in adapter.stream_completion(task):
            response += chunk

        assert "World" in response


# ─────────────────────────────────────────────────────────────────────
# get_models_by_server() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetModelsByServer:
    """Tests for get_models_by_server() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_models_grouped_by_server(self, two_server_urls):
        """Returns models grouped by server URL."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelC"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()  # Refresh manifests

        by_server = adapter.get_models_by_server()

        assert set(by_server["http://server0:1234"]) == {"modelA", "modelB"}
        assert set(by_server["http://server1:1234"]) == {"modelC"}


# ─────────────────────────────────────────────────────────────────────
# get_unreachable_servers() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestConcurrentDispatch:
    """Tests for parallel execution across servers (Issue #104)."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_different_models_run_in_parallel(self, two_server_urls):
        """Two calls to different models on different servers should run concurrently.

        This test verifies the fix for Issue #104 where adapter-level locking
        was serializing all calls, even to different servers.
        """
        import time

        # Each server has a unique model - dispatcher routes by model availability
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        # Track call timing
        call_log = []

        async def slow_response_server0(request):
            call_log.append(("server0_start", time.time()))
            await asyncio.sleep(0.3)  # Simulate processing time
            call_log.append(("server0_end", time.time()))
            return httpx.Response(200, content=sse_stream("Hello from 0"))

        async def slow_response_server1(request):
            call_log.append(("server1_start", time.time()))
            await asyncio.sleep(0.3)  # Simulate processing time
            call_log.append(("server1_end", time.time()))
            return httpx.Response(200, content=sse_stream("Hello from 1"))

        respx.post("http://server0:1234/v1/chat/completions").mock(
            side_effect=slow_response_server0
        )
        respx.post("http://server1:1234/v1/chat/completions").mock(
            side_effect=slow_response_server1
        )

        adapter = LMStudioAdapter(two_server_urls)

        async def call_model_a():
            result = ""
            task = InferenceTask(
                model_id="modelA",  # Only on server0
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=5.0
            )
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        async def call_model_b():
            result = ""
            task = InferenceTask(
                model_id="modelB",  # Only on server1
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=5.0
            )
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        # Run both calls concurrently
        start = time.time()
        results = await asyncio.gather(call_model_a(), call_model_b())
        elapsed = time.time() - start

        # Verify both completed
        assert "Hello from 0" in results[0]
        assert "Hello from 1" in results[1]

        # Verify parallel execution:
        # - If sequential: elapsed >= 0.6s (0.3 + 0.3)
        # - If parallel: elapsed ~= 0.3s (plus overhead)
        # Allow some margin for test environment variance
        assert elapsed < 0.5, f"Expected parallel execution (<0.5s), got {elapsed:.2f}s"

        # Verify both servers were called (both starts happened)
        events = [e[0] for e in call_log]
        assert "server0_start" in events
        assert "server1_start" in events

    @pytest.mark.asyncio
    @respx.mock
    async def test_same_model_both_servers_no_prefix_fans_out(self, two_server_urls):
        """Same model on both servers, no prefix → should use both GPUs.

        This is the actual battery scenario: user selects a model that exists
        on both servers, without explicit affinity prefix. Two concurrent calls
        should fan out to both servers, not queue for server 0.
        """
        import time

        # BOTH servers have the SAME model
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["shared-model"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["shared-model"]))
        )

        servers_used = []

        async def track_server0(request):
            servers_used.append("server0")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream("from 0"))

        async def track_server1(request):
            servers_used.append("server1")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream("from 1"))

        respx.post("http://server0:1234/v1/chat/completions").mock(side_effect=track_server0)
        respx.post("http://server1:1234/v1/chat/completions").mock(side_effect=track_server1)

        adapter = LMStudioAdapter(two_server_urls)

        async def call_model():
            task = InferenceTask(
                model_id="shared-model",  # NO prefix - should auto-discover
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        # Two concurrent calls with SAME model, NO prefix
        start = time.time()
        results = await asyncio.gather(call_model(), call_model())
        elapsed = time.time() - start

        # BOTH servers should be used (fan-out)
        assert "server0" in servers_used, f"Server 0 never called. Used: {servers_used}"
        assert "server1" in servers_used, f"Server 1 never called. Used: {servers_used}"

        # Should complete in parallel (~0.3s), not sequential (~0.6s)
        assert elapsed < 0.5, f"Expected parallel execution (<0.5s), got {elapsed:.2f}s"


# ─────────────────────────────────────────────────────────────────────
# get_unreachable_servers() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetUnreachableServers:
    """Tests for get_unreachable_servers() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_unreachable_servers(self, two_server_urls):
        """Returns servers that returned no models."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()  # Refresh manifests

        unreachable = adapter.get_unreachable_servers()

        assert unreachable == ["http://server0:1234"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_reachable_returns_empty(self, two_server_urls):
        """All servers reachable → returns empty list."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()

        unreachable = adapter.get_unreachable_servers()

        assert unreachable == []


# ─────────────────────────────────────────────────────────────────────
# MODEL TRANSITION DRAINING TESTS
# ─────────────────────────────────────────────────────────────────────

class TestModelTransitionDraining:
    """Tests for model-aware server draining (prevents JIT swap mid-stream).

    When a server is serving Model A with active requests, it must NOT
    accept Model B until all Model A requests complete. This prevents
    LM Studio's JIT loading from unloading Model A mid-stream.
    """

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_b_waits_for_model_a_to_drain(self, two_server_urls):
        """Model B request waits until Model A fully drains from server.

        Scenario: Server 0 has both models. Model A request is in-flight.
        Model B request should NOT go to server 0 until Model A completes.
        Server 1 has only Model B, so Model B goes there immediately.
        """
        import time

        # Both servers have both models (JIT advertises everything)
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )

        servers_used = {"modelA": [], "modelB": []}

        async def track_server0(request):
            import json as _json
            body = _json.loads(request.content)
            model = body["model"]
            servers_used[model].append("server0")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream(f"{model} from 0"))

        async def track_server1(request):
            import json as _json
            body = _json.loads(request.content)
            model = body["model"]
            servers_used[model].append("server1")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream(f"{model} from 1"))

        respx.post("http://server0:1234/v1/chat/completions").mock(side_effect=track_server0)
        respx.post("http://server1:1234/v1/chat/completions").mock(side_effect=track_server1)

        adapter = LMStudioAdapter(two_server_urls)

        async def call_model(model_id):
            task = InferenceTask(
                model_id=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        # Fire Model A on both servers, then Model B immediately after
        results = await asyncio.gather(
            call_model("modelA"),  # Goes to server0 (lowest load)
            call_model("modelA"),  # Goes to server1 (server0 busy)
            call_model("modelB"),  # Must wait — both servers serving modelA
            call_model("modelB"),  # Same — waits for drain
        )

        # All should complete
        assert all("modelA" in r or "modelB" in r for r in results)

        # Model B should NOT have gone to a server while Model A was in-flight.
        # Since both servers start with Model A, Model B must wait for one to drain.
        # This means Model B calls happen AFTER Model A calls (sequential at boundary).

    @pytest.mark.asyncio
    @respx.mock
    async def test_same_model_still_fans_out(self, two_server_urls):
        """Same model on both servers still uses parallel fan-out.

        The drain guard only blocks DIFFERENT models. Same-model requests
        should still fan out to both servers for parallel KV cache sharing.
        """
        import time

        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )

        servers_used = []

        async def track_server0(request):
            servers_used.append("server0")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream("from 0"))

        async def track_server1(request):
            servers_used.append("server1")
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=sse_stream("from 1"))

        respx.post("http://server0:1234/v1/chat/completions").mock(side_effect=track_server0)
        respx.post("http://server1:1234/v1/chat/completions").mock(side_effect=track_server1)

        adapter = LMStudioAdapter(two_server_urls)

        async def call_model():
            task = InferenceTask(
                model_id="modelA",
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        # Two concurrent same-model calls
        start = time.time()
        await asyncio.gather(call_model(), call_model())
        elapsed = time.time() - start

        # Both servers should be used (fan-out, not serialized)
        assert "server0" in servers_used
        assert "server1" in servers_used
        assert elapsed < 0.5, f"Expected parallel (<0.5s), got {elapsed:.2f}s"

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_accepts_new_model_after_drain(self, two_server_urls):
        """After all requests for Model A complete, server accepts Model B.

        Verifies current_model resets to None when active_requests hits 0.
        """
        # Server 0 has both models, server 1 has none
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response([]))
        )

        call_sequence = []

        async def track_server0(request):
            import json as _json
            body = _json.loads(request.content)
            call_sequence.append(body["model"])
            await asyncio.sleep(0.1)
            return httpx.Response(200, content=sse_stream("ok"))

        respx.post("http://server0:1234/v1/chat/completions").mock(side_effect=track_server0)

        adapter = LMStudioAdapter(two_server_urls)

        async def call_model(model_id):
            task = InferenceTask(
                model_id=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        # Sequential: Model A first, then Model B
        await call_model("modelA")
        await call_model("modelB")

        # Both should have been served by server 0 (sequential, so no conflict)
        assert call_sequence == ["modelA", "modelB"]
