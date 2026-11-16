# CPU-PROFILE calibration baseline (measurement only)

Run: `python -m experiments.cpu_profile run --set calibration --reps 5 --workers 1 2 4`
at code `91bba8e`, clean tree (benchmark `v0.5` unchanged; scripted actor, no
inference). Calibration set only (14 tasks, 5 reps, 3 worker settings = 210
runs); the final set was not run. Raw records stay in the gitignored
`runs/cpu-profile/`.

This **supersedes the earlier baselines** (taken before the validity,
independent-reference and reference-isolation corrections). An intermediate run
of the same command on an uncommitted tree was discarded, not reported.

**Validity: `valid: true`.** Full matrix 210/210 completed; every child exited
0; every run matched the independent reference (workspace, installed hidden
tests, verification output; the reference's tests ran in their own bubblewrap
after the fixture digest matched the suite pin) and each of the 14 tasks has a
single digest across all repetitions and all three worker counts (digests are in
the run's `summary.json`). The workload split hashed to the registered
`8c07eaa9…dda4`. CPU spent 83 s of the 1,200 s budget (admission control; no
overshoot). **The machine was heavily shared: load average ≈ 33 at the start on
12 CPUs** (earlier baselines saw 6-8), so wall times are inflated and noisy;
CPU time is the steadier figure. Absolute numbers are indicative only.

| 1 worker, median / p95 | seconds |
|---|---|
| wall per attempt | 1.56 / 2.41 |
| CPU (user+sys, process tree) | 0.39 / 0.49 |
| agent phase (one command, sandbox included) | 0.61 / 0.76 |
| verification (hidden tests) | 0.20 / 0.69 |
| GPU-monitor overhead | 0.10 / 0.26 |
| outside the engine (start-up, imports, teardown) | 0.68 / 1.24 |

Peak RSS of the largest process ≈ 31 MiB.

Scaling (70 runs per setting): 2,164 → 4,147 → 5,934 verified workflows/hour at
1 / 2 / 4 workers (2.7× at 4). Under this load these are lower than the
earlier, quieter-machine figures (3.4k → 6.2k → 10.9k); a comparison of an
optimisation against this baseline must therefore be interleaved in time, never
against a stored number.

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
