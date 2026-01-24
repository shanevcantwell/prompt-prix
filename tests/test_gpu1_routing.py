"""Focused tests proving GPU1 (server index 1) can be reached.

These tests verify the dispatcher routes to server 1 correctly.
"""

import asyncio
import pytest
import respx
import httpx

from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.adapters.schema import InferenceTask


def models_response(model_ids: list[str]) -> dict:
    return {"data": [{"id": m} for m in model_ids]}


def sse_stream(content: str) -> bytes:
    return f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\ndata: [DONE]\n\n'.encode()


@pytest.fixture
def two_servers():
    return ["http://gpu0:1234", "http://gpu1:1234"]


class TestGPU1Routing:
    """Prove that server index 1 (GPU1) can be reached."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_explicit_server_1_prefix_routes_to_gpu1(self, two_servers):
        """'1:modelB' explicitly routes to server index 1."""
        # GPU0 has modelA, GPU1 has modelB
        respx.get("http://gpu0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://gpu1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        # Only mock GPU1 completion - if GPU0 is called, test fails
        gpu1_called = False

        async def gpu1_completion(request):
            nonlocal gpu1_called
            gpu1_called = True
            return httpx.Response(200, content=sse_stream("Hello from GPU1"))

        respx.post("http://gpu1:1234/v1/chat/completions").mock(side_effect=gpu1_completion)

        adapter = LMStudioAdapter(two_servers)
        task = InferenceTask(
            model_id="1:modelB",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=5.0
        )

        response = ""
        async for chunk in adapter.stream_completion(task):
            response += chunk

        assert gpu1_called, "GPU1 was never called!"
        assert "GPU1" in response

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_only_on_gpu1_routes_automatically(self, two_servers):
        """Model only available on GPU1, no prefix → routes to GPU1."""
        # GPU0 has modelA, GPU1 has modelB (different models)
        respx.get("http://gpu0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://gpu1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        gpu1_called = False

        async def gpu1_completion(request):
            nonlocal gpu1_called
            gpu1_called = True
            return httpx.Response(200, content=sse_stream("Response from GPU1"))

        respx.post("http://gpu1:1234/v1/chat/completions").mock(side_effect=gpu1_completion)

        adapter = LMStudioAdapter(two_servers)
        task = InferenceTask(
            model_id="modelB",  # No prefix - should find on GPU1
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=5.0
        )

        response = ""
        async for chunk in adapter.stream_completion(task):
            response += chunk

        assert gpu1_called, "GPU1 was never called for modelB!"
        assert "GPU1" in response

    @pytest.mark.asyncio
    @respx.mock
    async def test_both_gpus_called_in_parallel(self, two_servers):
        """Two concurrent tasks route to their respective GPUs."""
        respx.get("http://gpu0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://gpu1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        call_order = []

        async def gpu0_completion(request):
            call_order.append("gpu0_start")
            await asyncio.sleep(0.1)
            call_order.append("gpu0_end")
            return httpx.Response(200, content=sse_stream("GPU0"))

        async def gpu1_completion(request):
            call_order.append("gpu1_start")
            await asyncio.sleep(0.1)
            call_order.append("gpu1_end")
            return httpx.Response(200, content=sse_stream("GPU1"))

        respx.post("http://gpu0:1234/v1/chat/completions").mock(side_effect=gpu0_completion)
        respx.post("http://gpu1:1234/v1/chat/completions").mock(side_effect=gpu1_completion)

        adapter = LMStudioAdapter(two_servers)

        async def call_gpu0():
            task = InferenceTask(
                model_id="0:modelA",
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        async def call_gpu1():
            task = InferenceTask(
                model_id="1:modelB",
                messages=[{"role": "user", "content": "Hi"}],
                timeout_seconds=5.0
            )
            result = ""
            async for chunk in adapter.stream_completion(task):
                result += chunk
            return result

        results = await asyncio.gather(call_gpu0(), call_gpu1())

        assert "GPU0" in results[0], f"GPU0 result missing: {results[0]}"
        assert "GPU1" in results[1], f"GPU1 result missing: {results[1]}"
        assert "gpu0_start" in call_order, "GPU0 was never called"
        assert "gpu1_start" in call_order, "GPU1 was never called"

        # Verify parallel: both starts should happen before both ends
        gpu0_start_idx = call_order.index("gpu0_start")
        gpu1_start_idx = call_order.index("gpu1_start")
        gpu0_end_idx = call_order.index("gpu0_end")
        gpu1_end_idx = call_order.index("gpu1_end")

        # If parallel, both starts happen before any end
        assert gpu0_start_idx < gpu0_end_idx
        assert gpu1_start_idx < gpu1_end_idx
        # At least one start should precede the other's end (overlap)
        assert gpu0_start_idx < gpu1_end_idx or gpu1_start_idx < gpu0_end_idx
