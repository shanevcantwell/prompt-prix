#!/usr/bin/env python3
"""
Headless battery runner - bypasses Gradio UI entirely.

Usage:
    python scripts/run_battery.py examples/tool_competence_tests.json --random 2
    python scripts/run_battery.py examples/tool_competence_tests.json --random 2 --export results.json

Loads servers from environment (LM_STUDIO_SERVER_1, LM_STUDIO_SERVER_2, etc.)
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from prompt_prix.config import load_servers_from_env, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_SECONDS
from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.mcp.registry import register_adapter
from prompt_prix.benchmarks import CustomJSONLoader
from prompt_prix.battery import BatteryRunner, RunStatus


async def main():
    parser = argparse.ArgumentParser(description="Run battery tests headlessly")
    parser.add_argument("file", help="Path to benchmark JSON/JSONL file")
    parser.add_argument("--random", "-r", type=int, default=2,
                        help="Number of random models to pick per server (default: 2)")
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})")
    parser.add_argument("--max-tokens", "-m", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress")
    args = parser.parse_args()

    # Load servers from environment
    servers = load_servers_from_env()
    if not servers:
        print("❌ No servers configured. Set LM_STUDIO_SERVER_1, LM_STUDIO_SERVER_2, etc. in .env")
        sys.exit(1)

    print(f"📡 Servers: {', '.join(servers)}")

    # Create adapter and register
    adapter = LMStudioAdapter(server_urls=servers)
    register_adapter(adapter)

    # Fetch models per server
    print("🔍 Fetching models...")
    await adapter.get_available_models()
    models_by_server = adapter.get_models_by_server()

    if not any(models_by_server.values()):
        print("❌ No models found on any server")
        sys.exit(1)

    # Pick random models per server
    selected_models = []
    for server_url, models in models_by_server.items():
        if not models:
            print(f"  ⚠️  {server_url}: no models")
            continue
        n = min(args.random, len(models))
        picks = random.sample(models, n)
        selected_models.extend(picks)
        print(f"  ✓ {server_url}: picked {picks}")

    if not selected_models:
        print("❌ No models selected")
        sys.exit(1)

    print(f"\n🎯 Selected {len(selected_models)} models: {selected_models}")

    # Load benchmark file
    print(f"\n📂 Loading {args.file}...")
    try:
        tests = CustomJSONLoader.load(args.file)
    except Exception as e:
        print(f"❌ Failed to load benchmark: {e}")
        sys.exit(1)

    print(f"  ✓ Loaded {len(tests)} test cases")

    # Create runner
    runner = BatteryRunner(
        tests=tests,
        models=selected_models,
        temperature=0.0,  # Deterministic for evals
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        max_concurrent=1  # Serialized execution
    )

    print(f"\n🚀 Running battery ({len(tests)} tests × {len(selected_models)} models = {len(tests) * len(selected_models)} cells)...\n")

    # Run and track progress
    last_completed = 0
    async for state in runner.run():
        if args.verbose and state.completed_count > last_completed:
            # Show each completion
            for key, result in state.results.items():
                if result.status in [RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.SEMANTIC_FAILURE]:
                    symbol = result.status_symbol
                    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
                    print(f"  {symbol} {result.model_id} × {result.test_id} ({latency})")
            last_completed = state.completed_count
        elif not args.verbose:
            # Progress bar style
            pct = state.progress_percent
            print(f"\r  Progress: {state.completed_count}/{state.total_count} ({pct:.0f}%)", end="", flush=True)

    print("\n")

    # Summary
    final_state = runner.state
    completed = sum(1 for r in final_state.results.values() if r.status == RunStatus.COMPLETED)
    semantic_fail = sum(1 for r in final_state.results.values() if r.status == RunStatus.SEMANTIC_FAILURE)
    errors = sum(1 for r in final_state.results.values() if r.status == RunStatus.ERROR)

    print("=" * 60)
    print(f"✅ Completed: {completed}")
    print(f"⚠️  Semantic failures: {semantic_fail}")
    print(f"❌ Errors: {errors}")
    print("=" * 60)

    # Print grid
    print("\nResults Grid:")
    grid = final_state.to_grid()
    print(grid.to_string(index=False))

    # Export if requested
    if args.export:
        export_data = {
            "tests": final_state.tests,
            "models": final_state.models,
            "results": []
        }
        for test_id in final_state.tests:
            for model_id in final_state.models:
                result = final_state.get_result(test_id, model_id)
                if result:
                    export_data["results"].append({
                        "test_id": result.test_id,
                        "model_id": result.model_id,
                        "status": result.status.value,
                        "response": result.response,
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                        "failure_reason": result.failure_reason
                    })

        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n📁 Exported to {args.export}")


if __name__ == "__main__":
    asyncio.run(main())
