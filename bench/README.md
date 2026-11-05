# Agentic coding benchmark

Measures whether a local model can complete real engineering work on a fixed
VRAM budget — not whether it wins a vendor benchmark.

## What it measures

Per task, `bench.py` runs a sandboxed agent loop against a disposable git repo,
then scores the result with hidden tests the agent never sees.

### Telemetry & Inference Metrics
- **Per-request streaming metrics**: `prompt_eval_count`, `prompt_eval_cached_count`,
  `prompt_eval_duration_s`, `eval_count`, `eval_duration_s`, `load_duration_s`,
  `total_duration_s`, and `done_reason`.
- **Latency & Throughput**: True TTFT (Time To First Token) separately tracking
  thinking start, content start, and tool call emission; decode throughput
  (`eval_count / eval_duration`) and prompt processing throughput
  (`prompt_eval_count / prompt_eval_duration`).
- **Context & Output Tracking**: Actual prompt tokens vs allocated `num_ctx`,
  cache hit ratio, and output truncation (`done_reason == "length"`).

### GPU & System Profiling
- **Device & Process Memory**: Device-wide VRAM and dedicated Ollama runner process
  VRAM (`/usr/lib/ollama/llama-server`) tracked via continuous background monitor.
- **Offload Ratio**: Automatic verification of layer offloading via `/api/ps`.
- **Status Reporting**: Explicit GPU telemetry states (`available`, `unavailable`, `unsupported`).

### Two-Axis Failure Taxonomy
Every task execution is evaluated across two orthogonal axes with empirical evidence:
- **Outcome** (what physically occurred): `passed`, `hidden_test_failure`, `invalid_action`,
  `timeout`, `repeated_command`, `premature_completion`, `forbidden_file_change`,
  `inference_failure`, `memory_failure`.
- **Possible Cause** (why it occurred, evidence-backed): `insufficient_reasoning`,
  `quantization_degradation`, `context_loss`, `tool_format_incompatibility`,
  `prompt_mismatch`, `test_fixture_defect`, `resource_constraint`, or `unknown`.
  Causes default strictly to `unknown` unless supported by verifiable trace metrics.

## Actuator Arms

- **Arm A (`--actuator text`)**: SWE-agent style text parsing with `RUN: <bash command>`
  and `DONE`. Supports heredocs, fenced blocks, and XML action wrappers.
- **Arm B (`--actuator native_tools`)**: Native JSON tool calling via `/api/chat` tools schema:
  - `execute_command(command: string)`: Run bash command in the sandbox.
  - `finish_task()`: Declare work complete.

## Sampling Configurations

Benchmarking supports reproducible sampling configurations:
- `--sampling-profile deterministic-v1` (default: `temp=0.0`, `seed=42`, `top_p=0.95`, `repeat_penalty=1.1`)
- Custom flags: `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, `--seed`

## Tasks

24 hand-written tasks in `tasks/`, each with:

- `repo/` — starting repository, stub or buggy code
- `tests/` — hidden tests, copied only into clean scoring workspace as `_hidden/`
- `solution/` — golden solution, used by `selfcheck.py`
- `meta.json` — category, prompt, test command, timeouts, allowed files

Categories: backend, debugging, devops, frontend, mobile, refactoring.
Tasks are independent, offline, and use only standard library / hermetic tools.
Each task run records a SHA256 `task_digest` fingerprinting task fixtures for reproducibility.

## Sandbox & Isolation

Execution runs inside `bwrap` with a read-only root, tmpfs `/home` and `/root`,
a private `/tmp`, the worktree bound at `/mnt/work`, and `--unshare-net`.

- **Strict Isolation (`--strict-isolation`)**: Enforces Linux cgroup v2 memory limits
  (fails closed if cgroups are unavailable or misconfigured).
- **Tamper-Proof Scoring**: The scorer rebuilds a pristine workspace via `clean_copy`
  (stripping symlinks and FIFOs), injects hidden tests with importable package markers,
  and compares changes against pre-run `baseline.json`.

## Usage

```bash
# List available tasks
uv run python bench/bench.py --list

# Run single task with deterministic sampling and streaming telemetry
uv run python bench/bench.py --model qwen3.5:4b --ctx 32768 --ngpu 99 --tasks debugging-default-arg

# Compare actuators (Arm A vs Arm B)
uv run python bench/bench.py --model qwen3.5:4b --actuator text --tag text-arm
uv run python bench/bench.py --model qwen3.5:4b --actuator native_tools --tag tools-arm

# Resume interrupted run
uv run python bench/bench.py --resume runs/20261105-xxxx-tag

# Run full harness selfcheck (regression tests, integration tests, golden checks)
uv run python bench/selfcheck.py
```

Results land in `runs/<stamp>-<tag>/` with `results.json`, `env.json`
(versions, driver, power limit, model digests, Ollama unit environment) and one
`transcript.json` per task.
