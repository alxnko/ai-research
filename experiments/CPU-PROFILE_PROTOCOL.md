# CPU-PROFILE: where the benchmark's own CPU-side time goes

Status: protocol for **measurement only**. No optimisation is registered here.
Separate from CPU-only LLM inference (a different experiment, not covered).

## Question
Of the wall time and CPU a verified attempt costs *outside* model inference
(process start-up, run configuration, workspace preparation, sandbox and command
execution, GPU-telemetry sampling, hidden-test verification, result writing),
which parts are large enough to be worth optimising?

## Hypothesis (to be tested later, not here)
CPU-side orchestration has measurable overhead that can be reduced without
changing any deterministic output.

## Workload
- Suite `v0.5`; the 24 pinned tasks are split once, by rule, in
  `experiments/cpu_workload.json` (sha256 `8c07eaa9e154a24398144e415c92af3f55adc55bfc6ee93615f53023d028dda4`): sorted by id within category,
  even positions → **calibration** (14 tasks), odd → **final** (10 tasks).
- A **scripted actor** replaces the model through the engine's own mock-model
  hook: one `execute_command` that writes the task's reference solution, then
  `finish_task`. Everything else is the unchanged `v0.5` engine, run as
  `main()` in a fresh process per repetition under strict isolation. It is a
  load generator: no result says anything about any model.
- No Ollama inference. The engine still samples GPU telemetry (nvidia-smi
  queries) as in a real run; that is part of what is measured.

## Measurements (per repetition, from outside the process)
wall time; user and system CPU of the whole process tree (`wait4`); peak
resident set of the largest process in it. From the engine's own row: agent,
verification and monitor-overhead seconds, and the engine's total, so
*outside-engine* time (start-up, imports, teardown) is wall − engine total.
Per group: median, p95 (nearest rank), min, max, execution failures. Batch
level: workflows per hour at 1, 2 and 4 workers (never more than 4).

## Correctness gate
Every repetition must solve its task (`passed` and `run_successful`); a
failure is an *execution failure* and fails the run. Each repetition also yields
a canonical digest of its deterministic outputs (task, fixture digest, outcome,
changed/added/removed files, commands, turns, tool calls, tokens). Repetitions
of a task must agree. The summary stores the per-task reference digests; any
later, optimised version must reproduce them exactly. Faster by doing less is
not an optimisation.

## Procedure and budget
1. **Calibration** (this PR): `run --set calibration --reps 5 --workers 1 2 4`.
   Tooling is developed and tuned on this set only.
2. **Final** (not run here): reserved. `--final` is required, so it cannot be
   started by accident. It is measured only when a single-variable optimisation
   has been registered, as an interleaved baseline/optimised pair on the same
   machine state, with a stated minimum improvement and number of repetitions.
3. Budget: at most 20 minutes of CPU time in total (enforced by the tool: no run
   starts once the spent CPU reaches it, and the stop is recorded), at most 4
   concurrent workers, no inference, no GPU compute.
4. Stop rules: an execution failure or a non-reproducible digest stops
   interpretation until explained; a budget stop is reported, never hidden.

## Not allowed
Optimising against the final set; changing `bench/` execution semantics under
the `v0.5` name (that needs a new benchmark version and tag); counting a
speed-up that skips checks, verification or sandboxing; editing EXP-2,
SMOKE-LIVE-1 records or tags.

## Limits
Shared machine: other processes add load (recorded: load average at start), so
absolute numbers are indicative and comparisons must be paired and repeated.
Peak RSS is the largest single process, not the tree's sum. The scripted actor
exercises one command per task, not the long command sequences of real
attempts, so shell- and sandbox-start costs are weighted more than in practice.
