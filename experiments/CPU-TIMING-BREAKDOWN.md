# CPU timing breakdown (read-only diagnostic)

Status: plan registered **before** any measurement in this document was taken.
Tool: `experiments/cpu_breakdown.py`. Scope: the 14 calibration tasks only; the
reserved final set is not used. No model inference, no GPU work (the GPU monitor
is only *observed*), no change to `bench/` (byte-identical to `v0.5`), no
optimization. Raw records stay in the gitignored `runs/`.

## Question
Where does the ~1-1.5 s of CPU-side wall time per scripted attempt go, and which
part, if any, is a specific, reproducible overhead that could later be reduced
without weakening the benchmark's guarantees? The calibration baseline cannot
say: it only has phase totals.

## Method
Everything is observed from outside the frozen engine: the traced child replaces
`subprocess.Popen` and wraps engine functions *in the child process only*
(the same technique the scripted actor already uses). Timestamps are
`time.monotonic()`; each event also records its thread.

1. **Traced attempts.** For every calibration task, N repetitions, one worker,
   alternating with **plain** (untraced) attempts of the same task in an
   interleaved order, so instrumentation overhead and machine drift are
   measured, not assumed. Per attempt, on the critical (main-thread) path:
   - interpreter start-up (parent spawn -> first line of the child) and imports;
   - `prepare` (workspace + git set-up) and each `run_sandboxed` call split into:
     the cgroup probe (`check_cgroup_v2`, a `systemd-run` launch), the sandboxed
     run itself (scope creation + bubblewrap + command, with the command's own
     time as seen by the engine), and the post-run `systemctl` kill;
   - verification, the GPU monitor's `start`/`stop` (including how long `stop`
     blocks) and engine teardown;
   - every subprocess launch: argv head, spawn time (fork/exec) and lifetime.
2. **GPU monitor, observed.** Backend (NVML or `nvidia-smi`), sample count, time
   per sample (wall and thread CPU), samples that overlap the critical path.
   `monitor_overhead_s` is time spent in a background thread; it is **not** added
   to the critical path.
3. **Isolated micro-measurements**, variants interleaved in rotating order, to
   split what the traced run cannot: `systemd-run` scope probe; bubblewrap
   alone; scope + bubblewrap; `systemctl kill` of an absent unit; `python -I -c
   pass`; `python -c "import bench.bench"`; one monitor sample (both backends if
   available).
4. Every attempt must still pass the existing correctness gate (the same
   independent reference and digests as `cpu_profile`); a traced run whose digest
   differs is reported as a failure, not averaged in.
5. Host state is recorded per attempt: load average and CPU pressure
   (`/proc/pressure/cpu`). Attempts are split into a **low-load half and a
   high-load half** at the median pressure.

## Reporting rules (fixed in advance)
- Per component: median, p95, min/max and IQR over attempts, and its **share of
  the median critical-path wall**. Instrumentation overhead = median(traced) -
  median(plain) wall and CPU; if it exceeds 10% of plain wall, shares are
  flagged as inflated.
- A component is **confirmed dominant** only if all hold: its median share is at
  least 20%; the isolated micro-measurement reproduces its cost within a factor
  of 2; and it ranks the same in both the low-load and high-load halves.
- It is **plausible but unproven** if its share is at least 10% but any of the
  three conditions fails.
- If the low- and high-load halves disagree on the ranking of the top components,
  the study is **inconclusive due to load** and says so; it does not pick a
  winner.
- CPU budget for the whole diagnostic: 300 CPU-seconds, admission-controlled like
  `cpu_profile`. At most 1 concurrent attempt (the breakdown wants an
  uncontended critical path; contention is already covered by the baseline).

## Out of scope / guard rails
- No optimization is implemented; this document registers no treatment.
- The reference-sandbox root bind (`--ro-bind / /`) is a known limitation for
  untrusted fixtures and is deliberately not touched here, as that would change
  the measured environment.
- A faster result obtained by skipping work never counts; this study changes
  nothing.
