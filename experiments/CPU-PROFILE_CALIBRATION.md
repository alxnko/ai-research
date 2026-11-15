# CPU-PROFILE calibration baseline (measurement only)

Run: `python -m experiments.cpu_profile run --set calibration --reps 5 --workers 1 2 4`
at code `983e59d` (benchmark `v0.5` unchanged; scripted actor, no inference).
Calibration set only (14 tasks, 5 reps, 3 worker settings = 210 runs); the final
set was not run. Raw records stay in the gitignored `runs/cpu-profile/`.

This **supersedes the first baseline** (taken at `5f6b6fd` before the validity
and independent-reference corrections; its numbers were within noise of these).

**Validity: `valid: true`.** Full matrix 210/210 completed; every child exited
0; every run matched the independent reference (workspace, installed hidden
tests, verification output) and the pinned fixture digest; each of the 14 tasks
has a single digest across all repetitions and all three worker counts (the
14 reference digests are in the run's `summary.json`). CPU spent 66 s of the
1,200 s budget (admission control; no overshoot). Machine shared, load average
≈ 6–8 at the start on 12 CPUs: absolute numbers are indicative.

| 1 worker, median / p95 | seconds |
|---|---|
| wall per attempt | 1.02 / 1.39 |
| CPU (user+sys, process tree) | 0.31 / 0.36 |
| agent phase (one command, sandbox included) | 0.57 / 0.59 |
| verification (hidden tests) | 0.09 / 0.49 |
| GPU-monitor overhead | 0.07 / 0.09 |
| outside the engine (start-up, imports, teardown) | 0.33 / 0.48 |

Peak RSS of the largest process ≈ 31 MiB.

Scaling (70 runs per setting): 3,361 → 6,238 → 10,870 verified workflows/hour
at 1 / 2 / 4 workers (3.2× at 4); per-run wall median 1.02 → 1.01 → 1.13 s,
monitor overhead 0.07 → 0.08 → 0.09 s. One worker-2 run took 3.0 s wall (agent
phase 2.2 s): a single outlier on a shared machine, retained.

## What this suggests (hypotheses, not findings)
These need a read-only timing breakdown before anything is changed:
- About 0.57 s of ~1.0 s is the agent phase for one trivial command, with only
  ~0.3 s of CPU in the whole attempt: mostly waiting (sandbox and cgroup
  set-up, fixed delays, polling?). Break it into sandbox creation, cgroup
  set-up, command execution and teardown.
- Process start-up and imports are a third of the wall time (0.33 s). Separate
  interpreter start, imports, initialisation and process launch.
- Two tasks dominate the verification tail (`backend-idempotency-v2`,
  `debugging-money-round-v2`).
- The GPU monitor costs ~7% of wall; check whether sampling frequency or
  subprocess launches dominate.
- Real attempts add seconds of model latency (SMOKE-LIVE-1: 18 s), so this
  overhead is ~1 s of ~19 s there.

Any optimisation must be registered separately, change one variable, and
reproduce the reference digests exactly, measured on the final set.
