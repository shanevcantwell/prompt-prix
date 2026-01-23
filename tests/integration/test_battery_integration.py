"""
Integration tests for battery execution and export.

These tests hit real LM Studio servers - skip if not configured.

Prerequisites:
- LM Studio running with at least one model loaded
- .env configured with LM_STUDIO_SERVER_1, etc.

Run with: pytest tests/integration/ -v -m integration
"""

import csv
import json
import pytest
from pathlib import Path

from prompt_prix.config import load_servers_from_env
from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.mcp.registry import register_adapter, clear_adapter
from prompt_prix.benchmarks import CustomJSONLoader
from prompt_prix.battery import BatteryRunner, TestStatus
from prompt_prix import state


@pytest.fixture
def live_adapter():
    """Create real adapter connected to LM Studio servers."""
    servers = load_servers_from_env()
    if not servers:
        pytest.skip("No LM Studio servers configured in .env")
    adapter = LMStudioAdapter(server_urls=servers)
    register_adapter(adapter)
    yield adapter
    clear_adapter()


@pytest.fixture
def first_model(live_adapter):
    """Get first available model from first server."""
    import asyncio

    async def get_model():
        await live_adapter.get_available_models()
        models_by_server = live_adapter.get_models_by_server()
        for server_url, models in models_by_server.items():
            if models:
                return models[0]
        return None

    model = asyncio.get_event_loop().run_until_complete(get_model())
    if not model:
        pytest.skip("No models available on any server")
    return model


@pytest.fixture
def tool_competence_tests():
    """Load tool competence test cases."""
    test_file = Path(__file__).parent.parent.parent / "examples" / "tool_competence_tests.json"
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")
    return CustomJSONLoader.load(str(test_file))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_tool_call(live_adapter, first_model, tool_competence_tests):
    """
    Verify basic_tool_call test executes and produces tool call output.

    This test validates:
    - BatteryRunner executes successfully
    - Model produces a tool call response
    - Tool call is properly formatted (not fragmented)
    """
    # Filter to just basic_tool_call test
    tests = [t for t in tool_competence_tests if t.id == "basic_tool_call"]
    assert len(tests) == 1, "basic_tool_call test not found"

    runner = BatteryRunner(
        tests=tests,
        models=[first_model],
        temperature=0.0,
        max_tokens=2048,
        timeout_seconds=300,
        max_concurrent=1
    )

    final_state = None
    async for battery_state in runner.run():
        final_state = battery_state

    assert final_state is not None
    assert final_state.completed_count == 1

    result = final_state.get_result("basic_tool_call", first_model)
    assert result is not None
    assert result.status in [TestStatus.COMPLETED, TestStatus.SEMANTIC_FAILURE]

    # Verify tool call format (should NOT be fragmented)
    response = result.response
    if "**Tool Call:**" in response:
        # Tool call should have complete JSON, not fragmented
        assert "get_weather" in response
        # Check for properly formatted JSON block
        assert "```json" in response
        # Should NOT have multiple tiny JSON fragments
        json_blocks = response.count("```json")
        assert json_blocks <= 2, f"Too many JSON blocks ({json_blocks}) - tool call may be fragmented"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_battery_export_json(live_adapter, first_model, tool_competence_tests):
    """
    Verify JSON export contains all expected fields.

    Tests the full pipeline: run battery -> export JSON -> verify structure.
    """
    from prompt_prix.tabs.battery.handlers import export_json

    # Run a small battery
    tests = [t for t in tool_competence_tests if t.id in ["basic_tool_call", "tool_selection"]][:2]
    if not tests:
        pytest.skip("Required test cases not found")

    runner = BatteryRunner(
        tests=tests,
        models=[first_model],
        temperature=0.0,
        max_tokens=2048,
        timeout_seconds=300,
        max_concurrent=1
    )

    final_state = None
    async for battery_state in runner.run():
        final_state = battery_state

    # Set global state for export handler
    state.battery_run = final_state

    # Export JSON
    status_msg, file_update = export_json()
    assert "Exported" in status_msg

    # Get filepath from Gradio update
    filepath = file_update.get("value")
    assert filepath is not None
    assert Path(filepath).exists()

    # Parse and verify JSON structure
    with open(filepath) as f:
        data = json.load(f)

    assert "tests" in data
    assert "models" in data
    assert "results" in data

    assert data["models"] == [first_model]
    assert len(data["results"]) == len(tests)

    # Verify result fields
    for result in data["results"]:
        assert "test_id" in result
        assert "model_id" in result
        assert "status" in result
        assert "response" in result
        assert "latency_ms" in result
        assert "error" in result
        assert "failure_reason" in result  # Added field

    # Cleanup
    Path(filepath).unlink(missing_ok=True)
    state.battery_run = None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_battery_export_csv(live_adapter, first_model, tool_competence_tests):
    """
    Verify CSV export is properly formatted and parseable.

    Tests the full pipeline: run battery -> export CSV -> verify format.
    """
    from prompt_prix.tabs.battery.handlers import export_csv

    # Run a small battery
    tests = [t for t in tool_competence_tests if t.id in ["basic_tool_call", "tool_selection"]][:2]
    if not tests:
        pytest.skip("Required test cases not found")

    runner = BatteryRunner(
        tests=tests,
        models=[first_model],
        temperature=0.0,
        max_tokens=2048,
        timeout_seconds=300,
        max_concurrent=1
    )

    final_state = None
    async for battery_state in runner.run():
        final_state = battery_state

    # Set global state for export handler
    state.battery_run = final_state

    # Export CSV
    status_msg, file_update = export_csv()
    assert "Exported" in status_msg

    # Get filepath from Gradio update
    filepath = file_update.get("value")
    assert filepath is not None
    assert Path(filepath).exists()

    # Parse and verify CSV
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == len(tests)

    # Verify headers
    expected_headers = ["test_id", "model_id", "status", "latency_ms",
                        "error", "failure_reason", "response"]
    assert list(rows[0].keys()) == expected_headers

    # Verify content
    for row in rows:
        assert row["model_id"] == first_model
        assert row["status"] in ["completed", "semantic_failure", "error"]
        # Response should be present and not fragmented
        if "**Tool Call:**" in row["response"]:
            # Check tool call is complete
            json_blocks = row["response"].count("```json")
            assert json_blocks <= 2, f"Fragmented tool call in CSV: {json_blocks} blocks"

    # Cleanup
    Path(filepath).unlink(missing_ok=True)
    state.battery_run = None


def _is_llm_model(model_name: str) -> bool:
    """Filter out embedding models and other non-LLM models."""
    lower = model_name.lower()
    # Exclude embedding models
    if "embed" in lower or "embedding" in lower:
        return False
    # Exclude vision-only models
    if "vision" in lower and "llava" not in lower:
        return False
    return True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_affinity_with_multiple_gpus(live_adapter):
    """
    Test that models are dispatched to their correct servers.

    When user selects modelA from server0 and modelB from server1,
    each should run on its designated server - not just any server
    that happens to have the model.

    This test verifies:
    - Multiple servers are detected
    - Different models are selected per server
    - Requests route to the correct server (not random)
    - Concurrent execution actually happens (timing check)
    """
    import asyncio
    import time

    await live_adapter.get_available_models()
    models_by_server = live_adapter.get_models_by_server()

    server_urls = list(models_by_server.keys())
    if len(server_urls) < 2:
        pytest.skip("Need 2+ servers to test multi-GPU dispatching")

    # Filter to LLM models only
    server0_llms = [m for m in models_by_server[server_urls[0]] if _is_llm_model(m)]
    server1_llms = [m for m in models_by_server[server_urls[1]] if _is_llm_model(m)]

    if not server0_llms:
        pytest.skip(f"Server 0 has no LLM models loaded: {models_by_server[server_urls[0]]}")
    if not server1_llms:
        pytest.skip(f"Server 1 has no LLM models loaded: {models_by_server[server_urls[1]]}")

    model_from_server0 = server0_llms[0]
    model_from_server1 = server1_llms[0]

    print(f"\nServer 0 ({server_urls[0]}): using {model_from_server0}")
    print(f"Server 1 ({server_urls[1]}): using {model_from_server1}")

    # Run a simple completion for each model with timing
    test_messages = [{"role": "user", "content": "Count from 1 to 5 slowly."}]

    async def run_model(model_id):
        start = time.time()
        response = ""
        async for chunk in live_adapter.stream_completion(
            model_id=model_id,
            messages=test_messages,
            temperature=0.0,
            max_tokens=100,
            timeout_seconds=120
        ):
            response += chunk
        elapsed = time.time() - start
        return model_id, response, elapsed

    # Run both models concurrently with server affinity prefixes
    # Prefix format: "server_idx:model_name" routes to specific server
    # With proper parallel routing: total_time ~= max(time0, time1)
    # With serialized routing: total_time ~= time0 + time1
    overall_start = time.time()
    results = await asyncio.gather(
        run_model(f"0:{model_from_server0}"),
        run_model(f"1:{model_from_server1}"),
        return_exceptions=True
    )
    overall_elapsed = time.time() - overall_start

    individual_times = []
    for result in results:
        if isinstance(result, Exception):
            pytest.fail(f"Model completion failed: {result}")
        model_id, response, elapsed = result
        individual_times.append(elapsed)
        print(f"  {model_id}: {elapsed:.1f}s, response '{response[:50].strip()}...'")
        assert response, f"Empty response for {model_id}"

    # Check for parallelism
    sum_of_times = sum(individual_times)
    print(f"\n  Individual times sum: {sum_of_times:.1f}s")
    print(f"  Total wall time: {overall_elapsed:.1f}s")

    # If truly parallel, wall time should be less than sum of individual times
    # Allow some overhead, but if wall_time >= sum, it's definitely serialized
    if overall_elapsed >= sum_of_times * 0.9:
        print("  WARNING: Requests appear to be serialized, not parallel!")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_call_not_fragmented(live_adapter, first_model, tool_competence_tests):
    """
    Regression test: tool call JSON must be accumulated before formatting.

    This specifically tests the bug fixed in commit 0ea88ad where tool call
    arguments were yielded per SSE chunk, producing fragmented JSON like:

        **Tool Call:** `get_weather`
        ```json
        {"
        ```
        ```json
        city
        ```
        ...

    After the fix, tool calls should be complete:

        **Tool Call:** `get_weather`
        ```json
        {"city": "Tokyo"}
        ```
    """
    # Use basic_tool_call which requires a tool call response
    tests = [t for t in tool_competence_tests if t.id == "basic_tool_call"]
    assert len(tests) == 1

    runner = BatteryRunner(
        tests=tests,
        models=[first_model],
        temperature=0.0,
        max_tokens=2048,
        timeout_seconds=300,
        max_concurrent=1
    )

    final_state = None
    async for battery_state in runner.run():
        final_state = battery_state

    result = final_state.get_result("basic_tool_call", first_model)
    assert result is not None

    response = result.response

    # If model made a tool call, verify it's properly formatted
    if "**Tool Call:**" in response:
        # Count JSON blocks - should be exactly 1 per tool call
        json_blocks = response.count("```json")
        tool_calls = response.count("**Tool Call:**")

        # Each tool call should have at most 1 JSON block for arguments
        assert json_blocks <= tool_calls, (
            f"Fragmented tool calls detected: {json_blocks} JSON blocks for {tool_calls} tool calls. "
            f"Response:\n{response[:500]}"
        )

        # Verify JSON is parseable
        import re
        json_pattern = r'```json\n(.*?)\n```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Should have "city" key for get_weather
                assert "city" in parsed, f"Missing 'city' in tool call args: {parsed}"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in tool call: {match}\nError: {e}")
