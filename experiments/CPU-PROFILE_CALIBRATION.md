# CPU-PROFILE calibration baseline (measurement only)

Run: `python -m experiments.cpu_profile run --set calibration --reps 5 --workers 1 2 4`
at code `5f6b6fd` (benchmark `v0.5` unchanged; scripted actor, no inference).
Calibration set only (14 tasks, 5 reps, 3 worker settings = 210 runs); the final
set was not run. Raw records stay in the gitignored `runs/cpu-profile/`.

Correctness: 210/210 solved, 0 execution failures, every task reproduced a
single canonical digest across all repetitions and worker counts (14 reference
digests). CPU spent 69 s of the 1,200 s budget. Machine shared, load average
≈ 11–14 at the start (12 CPUs): treat absolute numbers as indicative.

| 1 worker, median / p95 | seconds |
|---|---|
| wall per attempt | 1.03 / 1.56 |
| CPU (user+sys, process tree) | 0.30 / 0.35 |
| agent phase (one command, sandbox included) | 0.56 / 0.58 |
| verification (hidden tests) | 0.13 / 0.53 |
| GPU-monitor overhead | 0.06 / 0.08 |
| outside the engine (start-up, imports, teardown) | 0.34 / 0.55 |

Peak RSS of the largest process ≈ 31 MiB.

Scaling (70 runs per setting): 3,246 → 6,088 → 9,829 verified workflows/hour at
1 / 2 / 4 workers (3.0× at 4); per-run wall median rises 1.03 → 1.07 → 1.26 s,
CPU 0.30 → 0.30 → 0.39 s, monitor overhead 0.06 → 0.06 → 0.11 s.

## What this suggests (hypotheses, not findings)
- About 0.56 s of ~1.0 s is the agent phase for a single trivial command, with
  only ~0.3 s of CPU in the whole attempt: the attempt is mostly waiting
  (sandbox/cgroup set-up, fixed delays, polling). Worth locating before
  anything else.
- Process start-up and import cost is a third of the wall time (0.34 s).
- Verification is cheap for most tasks (0.13 s); two tasks dominate its tail
  (`backend-idempotency-v2` 0.53 s, `debugging-money-round-v2` 0.44 s).
- The GPU monitor costs ~6% of wall at one worker and roughly doubles at four.
- Real attempts add model latency of seconds (SMOKE-LIVE-1: 18 s), so this
  overhead is ~1 s of ~19 s there; the win is largest for many short tasks and
  for scripted/validation sweeps.

Any optimisation must be registered separately, change one variable, and
reproduce the reference digests exactly, measured on the final set.
