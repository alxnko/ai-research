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

## Correctness gate and validity
A repetition counts only if all of these hold:
1. the child process exited 0 and was not killed by the per-run timeout
   (300 s; a killed run is recorded, never dropped);
2. the engine reports the task solved;
3. three things measured **on disk** match an *independent reference* built by
   the tool from the pinned fixtures, without the engine's scorer or sandbox:
   the produced workspace (tree digest) equals the pristine repo with the
   reference solution applied; the hidden tests installed in the scoring copy
   equal the task's pinned tests; and the engine's verification output equals
   (timings removed) the output of one independent run of the task's test
   command on that reference. The task fixture digest is compared with the suite
   pin **before** any of its code runs, and that independent test run happens in
   its own bubblewrap (no network, empty environment, host files read-only, only
   the scratch copy writable), never on the host; without `bwrap` the reference
   fails closed;
4. every deterministic field is present (a missing or null field is a failure,
   never hashed as `None`).

The run specification is enforced: `--reps` and `--workers` must be positive
(workers at most 4), the workload split must hash to the sha256 registered
above (otherwise exit 2 before anything runs), and an invocation with no
attempts, or a matrix that is not tasks x reps as requested, is invalid.

Each passing repetition yields a canonical digest over those fields plus the
engine's outcome, file lists, commands, turns, tool calls and tokens. The
**invocation is valid only if** the whole requested matrix completed (every
task x repetition x worker count), no repetition failed, every task has exactly
one digest across all repetitions *and all worker counts*, the CPU budget was
not exceeded, and the independent reference itself passed. Otherwise the tool
exits non-zero, `summary.json` says `valid: false` and lists the problems, and
the raw measurements are kept. A failed attempt keeps its run directory and
child log under `<out>/failed/`.

What this does and does not show: an optimisation that skipped part of
verification would change the verification output or the installed hidden
tests and be caught; one that left the workspace and verification evidence
identical and merely ran faster passes, which is the intent. It does not prove
the sandbox's isolation properties are unchanged (those have their own tests).

## Procedure and budget
1. **Calibration** (this PR): `run --set calibration --reps 5 --workers 1 2 4`.
   Tooling is developed and tuned on this set only.
2. **Final** (not run here): reserved. `--final` is required, so it cannot be
   started by accident. It is measured only when a single-variable optimisation
   has been registered, as an interleaved baseline/optimised pair on the same
   machine state, with a stated minimum improvement and number of repetitions.
3. Budget: at most 20 minutes of CPU time, at most 4 concurrent workers, no
   inference, no GPU compute. The CPU budget is **admission control, not a hard
   cap**: before each wave the tool stops if the CPU already spent plus the
   worst per-run CPU seen so far times the wave size would reach it. A wave in
   flight is never killed (that would discard evidence), so overshoot is bounded
   by one wave; any overshoot is reported as a problem and makes the invocation
   invalid. The budget covers the measured benchmark process trees, not the
   tool's own hashing and the independent reference runs. `--cpu-budget` is
   limited to (0, 1200].
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
