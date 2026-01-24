"""
Stress tests for dispatch concurrency in LMStudioAdapter.
"""
import asyncio
import pytest
import respx
from httpx import Response
from prompt_prix.adapters.lmstudio import LMStudioAdapter

@pytest.mark.asyncio
@respx.mock
async def test_concurrent_dispatch_race_condition():
    """
    Reproduces race condition where multiple tasks see the same server as 'available'
    before one acquires it, causing unnecessary serialization.
    
    Setup:
    - 2 servers available, both have "model-A".
    - 2 concurrent requests for "model-A".
    
    Expected behavior (FIXED):
    - Both requests pick different servers (or retry efficiently) and run in parallel.
    - Duration should be approx equal to the single request duration (~0.5s).
    
    Actual behavior (RACE CONDITION):
    - Both tasks pick server1 because it looks free to both.
    - Task 1 acquires server1.
    - Task 2 waits on server1 lock.
    - Task 2 runs getting serialized after Task 1.
    - Duration is double (~1.0s).
    """
    server_urls = ["http://server1:1234", "http://server2:1234"]
    model_id = "model-A"
    
    # Define an async generator to simulate a slow stream (0.5s delay)
    async def delayed_stream():
        await asyncio.sleep(0.5)
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield b'data: [DONE]\n\n'
    
    # Mock both servers to have the model and handle chat completions
    for url in server_urls:
        respx.get(f"{url}/v1/models").mock(
            return_value=Response(200, json={"data": [{"id": model_id}]})
        )
        respx.post(f"{url}/v1/chat/completions").mock(
             return_value=Response(200, content=delayed_stream(), headers={"Content-Type": "text/event-stream"})
        )

    adapter = LMStudioAdapter(server_urls)
    
    # Helper to consume the stream
    async def run_request():
        async for _ in adapter.stream_completion(
            model_id=model_id,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=10,
            timeout_seconds=5
        ):
            pass

    start_time = asyncio.get_running_loop().time()
    
    # Run 2 requests concurrently.
    # We use gather to run them at the same time.
    await asyncio.gather(run_request(), run_request())
    
    duration = asyncio.get_running_loop().time() - start_time
    
    print(f"Test duration: {duration:.2f}s")
    
    # If successful parallel dispatch: duration ~0.5s + overhead
    # If serialized due to race condition: duration ~1.0s + overhead
    # We assert efficient parallelism (< 0.9s)
    assert duration < 0.9, f"Concurrent dispatch failed (took {duration:.2f}s), expected < 0.9s"
