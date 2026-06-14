# agent-runtime-lab

A framework-free, test-driven Agent Runtime built from scratch.

This repository implements the core control plane behind an AI agent without LangChain, LangGraph, or hidden orchestration layers. The goal is to make every production-critical boundary explicit: state transitions, tool routing, schema validation, retries, concurrency limits, subprocess execution, trace events, and HTTP API contracts.

The project is intentionally small, but it is designed around the same engineering constraints that matter in financial and infrastructure-grade AI systems: deterministic control, bounded side effects, auditability, and failure containment.

## Project Status

This is an educational runtime lab and prototype, not a production-ready agent platform.

The purpose of this repository is to make agent runtime mechanics visible and testable: control flow, validation, tracing, retries, concurrency, API boundaries, and side-effect containment. It should be treated as a learning and architecture demonstration project.

Requires Python 3.11+. Tested locally with Python 3.12.

## Security Notice

This project demonstrates application-level guardrails such as tool allowlists, schema validation, path checks, subprocess timeouts, and structured error handling.

It is not a secure sandbox.

Do not run untrusted scripts through this project. Do not expose the FastAPI service directly to the public internet. Do not treat the subprocess runner as an isolation boundary. Production-grade execution of untrusted code requires operating-system-level isolation such as containers, cgroups, seccomp, gVisor, Firecracker, separate users, filesystem restrictions, CPU and memory quotas, and network egress controls.

## What This Builds

`agent-runtime-lab` contains:

- A deterministic agent loop with explicit `AgentState`.
- A `ToolRegistry` allowlist for tool routing.
- Pydantic v2 input and output validation for every tool.
- Structured `ToolCall`, `ToolResult`, `AgentError`, and `TraceEvent` models.
- Retry policy for transient tool failures.
- Async batch execution with semaphore backpressure and hard timeout.
- Idempotency protection for repeated write-like actions.
- Controlled subprocess execution with script allowlist and path boundary checks.
- FastAPI endpoints for ticket-based async run submission, polling, trace lookup, and cancellation.
- A pytest suite covering normal paths, validation boundaries, failure policy, async execution, subprocess safety, and API contracts.

## Architecture

```text
==================================================================================================
                                 AGENT-RUNTIME-LAB ARCHITECTURE
==================================================================================================

       [ Client / Frontend ] (submit task, receive ticket, poll status)
                 |
                 v
   +--------------------------------------------------------------+
   |                        FASTAPI API LAYER                     |  <-- DTO / network boundary
   |   POST /runs              Submit task -> return run_id        |
   |   GET  /runs/{id}         Poll status and final answer        |
   |   GET  /runs/{id}/trace   Fetch structured trace events       |
   |   POST /runs/{id}/cancel  Request active task cancellation    |
   +-----------------------------+--------------------------------+
                                 |
                   Async Task    v    In-Memory State DB
   +--------------------------------------------------------------+
   |                     AGENT RUNTIME ENGINE                     |  <-- domain control loop
   |                                                              |
   |     +--------------------------------------------------+     |
   |     |    Deterministic Control Loop                    |     |
   |     |                                                  |     |
   |     |    1. Step boundary check                        |     |
   |     |    2. Provider adapter call                      |     |
   |     |    3. Tool-call routing through registry          |     |
   |     |    4. Structured transition and trace recording   |     |
   |     +------------------------+-------------------------+     |
   |                              |
   |                              v
   |     +--------------------------------------------------+     |
   |     |            ASYNC / SYNC TOOL EXECUTOR            |  <-- execution boundary
   |     |                                                  |     |
   |     |    Concurrency throttle: asyncio.Semaphore        |     |
   |     |    Batch timeout:        asyncio.wait_for         |     |
   |     |    Schema validation:    Pydantic v2              |     |
   |     |    Retry policy:         transient failures only   |     |
   |     +------------------------+-------------------------+     |
   +------------------------------+-------------------------------+
                                  |
                    Execution     v    Strict allowlists
   +--------------------------------------------------------------+
   |                     ENVIRONMENT EXECUTORS                    |  <-- side-effect boundary
   |                                                              |
   |   Pure tools             Idempotent writes       Subprocess  |
   |   AddNumbersTool         Run marker guard        approved    |
   |                                                scripts only  |
   |                                                shell=False   |
   |                                                path boundary |
   |                                                hard timeout  |
   +--------------------------------------------------------------+
```

## Repository Layout

```text
agent-runtime-lab/
  pyproject.toml
  README.md
  src/
    agent_runtime/
      api.py
      config.py
      errors.py
      models.py
      tracing.py
      providers/
        base.py
        fake_provider.py
      runtime/
        async_executor.py
        loop.py
        state.py
      tools/
        arithmetic.py
        base.py
        idempotency_writer.py
        registry.py
        script_runner.py
  scripts/
    safe_describe_csv.py
    safe_series_summary.py
  sample_data/
    series/monthly_sales.csv
  tests/
    test_api.py
    test_async_executor.py
    test_error_policy.py
    test_registry.py
    test_runtime_loop.py
    test_script_runner.py
    test_validation.py
```

## Quick Start

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -e .
pip install pytest pytest-asyncio
```

Run the full test suite:

```bash
pytest -v
```

Expected result at the time of writing:

```text
22 passed
```

Run the FastAPI service:

```bash
uvicorn agent_runtime.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Submit a run:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"user_input":"Hello agent","max_steps":3}'
```

Poll a run:

```bash
curl http://127.0.0.1:8000/runs/{run_id}
```

Fetch trace:

```bash
curl http://127.0.0.1:8000/runs/{run_id}/trace
```

Cancel a run:

```bash
curl -X POST http://127.0.0.1:8000/runs/{run_id}/cancel
```

## Core Concepts

### 1. Deterministic Runtime Boundary

The LLM provider is probabilistic. The runtime is not.

The runtime owns:

- `run_id`
- message history
- `step_count`
- status transitions
- maximum step protection
- tool routing
- error classification
- trace recording

This separation matters because production software cannot rely on prompt instructions as a hard safety boundary. Prompt constraints are probabilistic. Code-level allowlists and validators are deterministic.

### 2. ToolRegistry as a Hard Allowlist

Tools are not discovered or executed by name freely. Every callable tool must be registered:

```python
reg = ToolRegistry()
reg.register(AddNumbersTool())
```

Unknown tool names return structured `unknown_tool` errors. Duplicate tool registration raises immediately instead of silently replacing existing behavior.

### 3. Pydantic as the Tool Boundary

Every tool declares:

- `input_model`
- `output_model`
- `run(args)`

`BaseTool.execute()` validates raw model-generated arguments before calling business logic and serializes validated output afterward. This catches:

- missing arguments
- invalid types
- domain rule violations
- malformed tool outputs

Example: `AddNumbersTool` rejects values whose magnitude exceeds the configured business boundary.

### 4. TraceEvent as the Audit Log

The runtime records structured events such as:

- `run_started`
- `model_requested`
- `model_responded`
- `tool_call_received`
- `tool_started`
- `tool_transient_error`
- `tool_succeeded`
- `tool_validation_failed`
- `tool_failed`
- `run_completed`
- `max_steps_exceeded`

This makes the agent observable. A failing run can be debugged by replaying the trajectory instead of guessing from the final answer.

### 5. Async Tool Execution and Backpressure

`AsyncToolExecutor` supports concurrent tool execution while preserving bounded resource usage:

- `asyncio.Semaphore` limits maximum concurrency.
- `asyncio.wait_for` enforces batch-level timeout.
- Timeout produces structured `timeout` results.
- Sync tools are executed through an executor so blocking work does not directly block the event loop.

This is a minimal model of backpressure for agent tool execution.

### 6. Idempotency for Write-Like Actions

`IdempotencyRunMarkerTool` demonstrates a basic guard against repeated side effects. It uses `operation_id` as a client-generated idempotency key and skips duplicate writes.

This is important because LLMs can retry, duplicate, or re-emit tool calls. Write-like tools must be protected against double execution.

### 7. Controlled Subprocess Execution

`ControlledScriptRunnerTool` intentionally treats subprocess execution as a dangerous side-effect boundary.

It enforces:

- approved script names only
- no arbitrary shell command strings
- `shell=False`
- argument arrays instead of command string concatenation
- path boundary under `sample_data/`
- hard subprocess timeout
- structured stdout, stderr, exit code, duration, and timeout result

Subprocess execution is useful for isolated analytics scripts, but it is not a complete sandbox.

## API Contract

### `GET /health`

Returns service health:

```json
{
  "status": "healthy",
  "service": "agent-runtime-core"
}
```

### `POST /runs`

Creates a run and immediately returns a ticket.

Request:

```json
{
  "user_input": "Hello agent",
  "max_steps": 3
}
```

Response:

```json
{
  "run_id": "...",
  "status": "running",
  "step_count": 0,
  "final_answer": null
}
```

### `GET /runs/{run_id}`

Returns the current run summary:

```json
{
  "run_id": "...",
  "status": "completed",
  "step_count": 1,
  "final_answer": "Processed via FastAPI Gateway seamlessly."
}
```

### `GET /runs/{run_id}/trace`

Returns the structured trace event list for a run.

### `POST /runs/{run_id}/cancel`

Requests cancellation for an active background task.

## Test Matrix

The test suite covers:

| Area | Coverage |
| --- | --- |
| Runtime loop | direct answer, single tool call, max step protection |
| Registry | duplicate registration rejection, schema listing |
| Validation | invalid types, missing args, business boundary, unknown tool |
| Error policy | transient retry self-healing, fatal unknown-tool stop |
| Async executor | concurrency speedup, semaphore limit, timeout cancellation, idempotency |
| Script runner | approved scripts, unknown script rejection, path traversal rejection, hard timeout |
| API | health check, create and poll lifecycle, trace endpoint, 404 boundary |

Run all tests:

```bash
pytest -v
```

Run one test file:

```bash
pytest tests/test_async_executor.py -v
```

Run one test:

```bash
pytest tests/test_error_policy.py::test_transient_error_self_healing -v
```

## Why Framework-Free

Agent frameworks are useful, but they can hide the exact control flow that matters most in financial and infrastructure-grade systems.

This lab intentionally avoids agent orchestration frameworks because it needs code-level ownership over:

- state machine transitions
- retry semantics
- step limits
- error classification
- trace event names and payloads
- tool allowlist enforcement
- concurrency and timeout behavior
- subprocess safety policy
- API DTO boundaries

The point is not that frameworks are bad. The point is that before delegating orchestration to a framework, the engineer should understand the runtime mechanics that make an agent safe.

## Security Model

This project uses application-level safety controls:

- registry allowlist for tools
- Pydantic schemas for inputs and outputs
- approved script list
- path boundary checks
- `shell=False`
- hard timeout
- structured error conversion

These controls reduce risk, but they are not a full operating-system sandbox.

Important distinction:

```text
Application allowlist:
  Decides what the program is allowed to attempt.

OS-level sandbox:
  Limits what a process can physically do even if the application is compromised.
```

Production-grade subprocess isolation should add controls such as containers, cgroups, seccomp, gVisor, Firecracker, separate users, filesystem mounts, CPU and memory quotas, and network egress policy.

## What This Is Not

- Not a LangChain or LangGraph replacement.
- Not a secure code execution sandbox.
- Not a production persistence layer.
- Not a real trading system.
- Not financial advice or an investment recommendation engine.
- Not intended to be exposed directly as a public internet service.

## Known Limitations

- The API uses an in-memory `RUNS_DATABASE`; state is lost on service restart.
- `ACTIVE_TASKS` is process-local and does not work across multiple server workers.
- The current API worker uses `FakeProvider`; real provider adapters are future work.
- The sync runtime is called from an async background task; a production version should use a fully async runtime or isolate blocking work.
- `subprocess.run(..., capture_output=True)` can be unsafe for very large outputs because stdout and stderr are buffered in parent process memory.
- The subprocess runner is allowlisted but not OS-sandboxed.
- Parallel tool execution currently assumes calls are independent; causal tool dependencies require explicit scheduling metadata.
- Trace output is returned directly; production APIs should filter sensitive payloads.

## Future Work

Next directions:

- Persistent run store using SQLite, DuckDB, Postgres, or Redis.
- Provider adapters for OpenAI-compatible APIs and local models.
- Context-window management and memory compaction.
- RAG and vector boundary management.
- MCP-compatible tool registration.
- Streaming API responses.
- Tool dependency graph for serial vs parallel scheduling.
- OS-level sandboxing for subprocess tools.
- Golden dataset evaluation for non-deterministic agent behavior.
- LLM-as-a-judge semantic assertions for answer quality and safety.

## Design Summary

The LLM is probabilistic. The runtime is deterministic.

This project demonstrates how to wrap a non-deterministic model inside hard software boundaries: typed messages, schema-validated tools, registry allowlists, bounded loops, structured errors, trace events, timeout controls, idempotency keys, and API DTOs.

That is the core engineering move: let the model propose actions, but let deterministic runtime code decide what is allowed to happen.

## License

MIT License. See [LICENSE](LICENSE).
