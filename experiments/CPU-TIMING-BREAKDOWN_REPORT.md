# CPU timing breakdown: results

Plan: `CPU-TIMING-BREAKDOWN.md` (registered before measuring). Tool:
`experiments/cpu_breakdown.py`, code `f8bab80`, clean tree, benchmark `v0.5`
unchanged. Calibration tasks only (14 tasks x 5 reps, traced and plain
interleaved = 140 attempts, one worker); the final set was not run; no GPU use
and no model. Raw records stay in the gitignored `runs/cpu-breakdown/`.

**Validity.** All 140 attempts matched the independent reference and each task
has one digest across traced and plain runs (tracing changes no output). CPU
spent 57 s of the 300 s diagnostic budget. **The host was heavily loaded**
(load average ~28 on 12 CPUs), so absolute times are inflated; the low-/high-
load split below is the check against that.

**Instrumentation overhead.** Traced 1.432 s vs plain 1.395 s median wall
(+0.038 s, 2.7%), CPU +0.003 s: below the 10% line, shares are not flagged as
inflated. The components close to the measured wall (within 0.1% in the trial).

## Where one scripted attempt's wall time goes
Critical path, main thread, exclusive of nested spans (median over 70 traced
attempts; share of the median critical path; low-/high-load halves split at the
median CPU pressure):

| component | median s | all | low load | high load |
|---|---|---|---|---|
| **GPU monitor `stop()`** | 0.515 | **36%** | 44% | 32% |
| sandbox launch + the command's own run (3 calls) | 0.271 | 19% | 19% | 19% |
| engine other (see below) | 0.265 | 18% | 14% | 21% |
| Python imports | 0.109 | 8% | 9% | 8% |
| cgroup probe (4 `systemd-run` launches) | 0.093 | 7% | 7% | 7% |
| engine teardown | 0.038 | 3% | 3% | 3% |
| cgroup kill (3 `systemctl` launches) | 0.028 | 2% | 2% | 2% |
| workspace prepare | 0.017 | 1% | 1% | 1% |
| interpreter start-up | 0.015 | 1% | 1% | 1% |

## Findings, classified by the registered rules

**Study-level outcome: INCONCLUSIVE DUE TO LOAD.** The plan fixes this rule in
advance: if the low- and high-load halves disagree on the ranking of the top
components, the study is inconclusive and does not pick a winner. They disagree
(top-3 order differs: the 2nd and 3rd places swap), so the registered outcome
of this study is *inconclusive*. This report does not declare a winner.

What can still be said, as an observation that the registered rules do not
promote to a conclusion: `VramMonitor.stop()` ranks first in both halves (36%
overall, 44% low load, 32% high load), the isolated measurement reproduces its
cost (0.508 s vs 0.515 s), and the mechanism is visible in the source
(`bench/telemetry.py`, not edited): the sampler loop ends each pass with
`time.sleep(self.interval_s)` (0.5 s) instead of waiting on its stop event, and
`stop()` sets the event and then joins the thread, so it waits for the sleep to
end. It is a blocking join, not CPU (about 2 ms CPU per sample). It meets the
per-component criteria for "confirmed dominant" (>= 20%, reproduced within 2x,
first in both halves) but, under the study-level rule, is a **strong candidate,
not a confirmed result**. The next step is therefore a registered, interleaved
control-vs-treatment comparison, which measures the effect directly and does not
depend on this ranking; a lower-load rerun of the breakdown is not required.

**Plausible but unproven**
- *Sandbox launch.* 19% (just under the 20% line), but most of it is the
  commands themselves running in the sandbox (the solution write, the hidden
  tests): isolated scope + bubblewrap + `true` costs 0.022 s per call, about
  0.07 s over the 3 calls (~5%). The launch overhead alone is below the 10% bar.
- *Engine "other" (18%).* Per attempt the main thread launches about 8-10
  short processes for environment metadata and monitor set-up (`nvidia-smi`
  x5, `ollama --version`, `ollama list`, `systemctl --user show`, `free`,
  `nproc`, `git`), ~0.28 s of process lifetime. Their ranking against the
  sandbox swaps between load halves, so the order of these two is
  **inconclusive due to load**.

**Not a bottleneck here**
- *The cgroup probe* (the leading hypothesis): 4 `systemd-run ... true` launches
  per attempt, 0.093 s, 7%. It fails the 10% bar (its in-place cost, 0.023 s per launch, is ~1.7x
  the isolated 0.013 s, which is within the 2x reproduction limit). Caching it
  could save at most ~6%.
- *`systemctl` kill* 2%; *interpreter start-up* 1%; *prepare* 1%.
- *Imports* 8% (about 0.11 s; `bench.telemetry`, `urllib`, `http.client`,
  `concurrent.futures`, `logging` lead). Not a candidate on its own.

**GPU monitor, observed.** No `pynvml`: every sample launches `nvidia-smi`
(~0.07-0.09 s wall, ~2 ms CPU) in the background thread, one sample per attempt
in this short workload. That time is overlapped, not added to the critical path
(it is not part of the table above); only `stop()`'s wait is.

## What this does not show
- **The 0.5 s is large only because these scripted attempts are short.** In a
  real attempt (SMOKE-LIVE-1: 18 s) the worst case is about 0.5 s (under 3%)
  and, since the join waits for the remainder of a sleep that started at a
  random point, the expected value is nearer 0.25 s. The benchmark's own phase
  timings in real runs were not re-measured here.
- Under this load, absolute and in-situ vs isolated numbers differ by ~2x; only
  the shares and the within-run comparisons are used.
- One host, one workload (14 calibration tasks, one command each); model
  latency and Ollama are absent by design.
- `bench/telemetry.py` is frozen at `v0.5`. **Removing the wait means editing a
  frozen file**, so it needs a new, versioned benchmark (or an explicit decision
  to treat it as a harness-only shim); this report registers nothing and changes
  nothing. An optimization that skips monitor work does not count: the sample
  count and peak-VRAM semantics must stay identical.
- The registered rule "if the halves disagree on the ranking of the top
  components, the study is inconclusive" is applied at the study level, as
  written (an earlier draft of this report applied it per component and
  declared a confirmed winner; that was wrong).
