# Implementation Plan

## Phase 1: Diagnosis and Stabilization
1.  **Verify Dispatcher Logic**: Run stress tests to confirm race conditions or inefficiencies in `LMStudioAdapter`.
2.  **Analyze `_ConcurrentDispatcher`**: Confirm if it is actually used and if it correctly handles concurrency.
3.  **Audit Resource Acquisition**: specific check on `find_and_acquire` atomicity.

## Phase 2: Refactoring
1.  **Integrate Dispatcher**: Ensure `BatteryRunner` and `CompareRunner` properly utilize the queue-based dispatcher (if they aren't already).
2.  **Type Safety**: Refactor `complete_stream` and adapter interfaces to use strong types (Pydantic models) instead of raw strings where appropriate.

## Phase 3: Validation
1.  **Stress Testing**: Enhance `tests/stress/test_dispatch_concurrency.py` to cover edge cases.
2.  **Integration Testing**: Verify end-to-end flow with mocked servers.
