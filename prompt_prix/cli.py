"""CLI entry point for prompt-prix.

Provides headless battery execution for agent consumption (via run_command)
and human terminal use. Same orchestration internals as the Gradio UI.

Entry point:
    prompt-prix-cli models [--json]
    prompt-prix-cli run-battery --tests <file> --models <ids> [-o results.json]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-prix-cli",
        description="Headless battery execution for prompt-prix.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # models
    models_p = sub.add_parser("models", help="List available models")
    models_p.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Full JSON output (models, servers, unreachable)",
    )

    # run-battery
    run_p = sub.add_parser("run-battery", help="Execute a battery test suite")
    run_p.add_argument("--tests", required=True, help="Benchmark file (JSON/JSONL/YAML)")
    run_p.add_argument("--models", required=True, help="Comma-separated model IDs")
    run_p.add_argument("--runs", type=int, default=1, help="Runs per cell (>1 = consistency)")
    run_p.add_argument("--drift-threshold", type=float, default=0.0, help="Cosine distance threshold")
    run_p.add_argument("--judge-model", default=None, help="Model ID for LLM-as-judge")
    run_p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    run_p.add_argument("--timeout", type=int, default=300, help="Timeout per request (seconds)")
    run_p.add_argument("-o", "--output", default=None, help="Output file path (default: stdout)")

    return parser


# ─────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────


async def _cmd_models(json_output: bool = False) -> int:
    """List available models. Returns exit code."""
    from prompt_prix.mcp.tools.list_models import list_models

    result = await list_models()

    if json_output:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for model_id in result["models"]:
            print(model_id)
        if result.get("unreachable"):
            print(
                f"\nUnreachable servers: {', '.join(result['unreachable'])}",
                file=sys.stderr,
            )

    return 0


def _load_tests(path: str):
    """Load benchmark cases from file, auto-detecting format."""
    from prompt_prix.benchmarks import CustomJSONLoader, PromptfooLoader

    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return PromptfooLoader.load(path)
    else:
        return CustomJSONLoader.load(path)


def _log_progress(
    state,
    prev_completed: set[str],
    counter: int,
    total: int,
    is_consistency: bool,
) -> tuple[set[str], int]:
    """Print one stderr line per newly completed result. Returns updated tracking."""
    from prompt_prix.battery import RunStatus

    if is_consistency:
        # Track individual run completions within aggregates
        for key, agg in state.aggregates.items():
            for i, result in enumerate(agg.results):
                run_key = f"{key}:run{i}"
                if run_key in prev_completed:
                    continue
                if result.status in (RunStatus.COMPLETED, RunStatus.SEMANTIC_FAILURE, RunStatus.ERROR):
                    prev_completed.add(run_key)
                    counter += 1
                    symbol = result.status_symbol
                    latency = f" {result.latency_ms / 1000:.1f}s" if result.latency_ms else ""
                    reason = f" ({result.failure_reason})" if result.failure_reason else ""
                    error = f" ({result.error})" if result.status == RunStatus.ERROR and result.error else ""
                    print(
                        f"[{counter}/{total}] {agg.model_id} | {agg.test_id} run {i + 1} "
                        f"{symbol}{latency}{reason}{error}",
                        file=sys.stderr,
                    )
    else:
        for key, result in state.results.items():
            if key in prev_completed:
                continue
            if result.status in (RunStatus.COMPLETED, RunStatus.SEMANTIC_FAILURE, RunStatus.ERROR):
                prev_completed.add(key)
                counter += 1
                symbol = result.status_symbol
                latency = f" {result.latency_ms / 1000:.1f}s" if result.latency_ms else ""
                reason = f" ({result.failure_reason})" if result.failure_reason else ""
                error = f" ({result.error})" if result.status == RunStatus.ERROR and result.error else ""
                print(
                    f"[{counter}/{total}] {result.model_id} | {result.test_id} "
                    f"{symbol}{latency}{reason}{error}",
                    file=sys.stderr,
                )

    return prev_completed, counter


def _export_battery(state) -> dict:
    """Build export dict from BatteryRun state (matches handlers.py format)."""
    export = {
        "tests": state.tests,
        "models": state.models,
        "results": [],
    }
    for test_id in state.tests:
        for model_id in state.models:
            result = state.get_result(test_id, model_id)
            if result:
                export["results"].append({
                    "test_id": result.test_id,
                    "model_id": result.model_id,
                    "status": result.status.value,
                    "response": result.response,
                    "latency_ms": result.latency_ms,
                    "judge_latency_ms": result.judge_latency_ms,
                    "error": result.error,
                    "failure_reason": result.failure_reason,
                })
    return export


def _export_consistency(state) -> dict:
    """Build export dict from ConsistencyRun state (matches handlers.py format)."""
    export = {
        "tests": state.tests,
        "models": state.models,
        "runs_total": state.runs_total,
        "aggregates": [],
    }
    for test_id in state.tests:
        for model_id in state.models:
            agg = state.get_aggregate(test_id, model_id)
            if agg:
                export["aggregates"].append({
                    "test_id": agg.test_id,
                    "model_id": agg.model_id,
                    "status": agg.status.value,
                    "passes": agg.passes,
                    "total": agg.total,
                    "avg_latency_ms": agg.avg_latency_ms,
                    "results": [
                        {
                            "status": r.status.value,
                            "response": r.response,
                            "latency_ms": r.latency_ms,
                            "error": r.error,
                            "failure_reason": r.failure_reason,
                        }
                        for r in agg.results
                    ],
                })
    return export


async def _cmd_run_battery(
    tests_path: str,
    models_csv: str,
    runs: int = 1,
    drift_threshold: float = 0.0,
    judge_model: Optional[str] = None,
    max_tokens: int = 2048,
    timeout: int = 300,
    output: Optional[str] = None,
) -> int:
    """Execute a battery run. Returns exit code."""
    from prompt_prix.mcp.tools.list_models import list_models

    # Load tests
    tests_file = Path(tests_path)
    if not tests_file.exists():
        print(f"Error: test file not found: {tests_path}", file=sys.stderr)
        return 1

    try:
        tests = _load_tests(tests_path)
    except Exception as e:
        print(f"Error loading tests: {e}", file=sys.stderr)
        return 1

    if not tests:
        print("Error: no test cases found in file", file=sys.stderr)
        return 1

    # Parse and validate models
    model_ids = [m.strip() for m in models_csv.split(",") if m.strip()]
    if not model_ids:
        print("Error: no models specified", file=sys.stderr)
        return 1

    result = await list_models()
    available = set(result["models"])
    missing = [m for m in model_ids if m not in available]
    if missing:
        print(f"Error: models not available: {', '.join(missing)}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(available))}", file=sys.stderr)
        return 1

    # Create runner
    is_consistency = runs > 1
    if is_consistency:
        from prompt_prix.consistency import ConsistencyRunner
        runner = ConsistencyRunner(
            tests=tests,
            models=model_ids,
            runs=runs,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            judge_model=judge_model,
            drift_threshold=drift_threshold,
        )
        total = len(tests) * len(model_ids) * runs
    else:
        from prompt_prix.battery import BatteryRunner
        runner = BatteryRunner(
            tests=tests,
            models=model_ids,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            judge_model=judge_model,
            drift_threshold=drift_threshold,
        )
        total = len(tests) * len(model_ids)

    # Execute with progress streaming
    print(
        f"Running {len(tests)} tests × {len(model_ids)} models"
        + (f" × {runs} runs" if is_consistency else ""),
        file=sys.stderr,
    )

    prev_completed: set[str] = set()
    counter = 0
    final_state = None

    async for state in runner.run():
        final_state = state
        prev_completed, counter = _log_progress(
            state, prev_completed, counter, total, is_consistency
        )

    if final_state is None:
        print("Error: runner produced no state", file=sys.stderr)
        return 1

    # Export
    if is_consistency:
        export_data = _export_consistency(final_state)
    else:
        export_data = _export_battery(final_state)

    if output:
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Results written to {output}", file=sys.stderr)
    else:
        json.dump(export_data, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(message)s", stream=sys.stderr)

    # Load env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Register adapters from environment
    from prompt_prix.mcp.registry import register_default_adapter
    register_default_adapter()

    # Dispatch
    if args.command == "models":
        code = asyncio.run(_cmd_models(json_output=args.json_output))
    elif args.command == "run-battery":
        code = asyncio.run(_cmd_run_battery(
            tests_path=args.tests,
            models_csv=args.models,
            runs=args.runs,
            drift_threshold=args.drift_threshold,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            output=args.output,
        ))
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
