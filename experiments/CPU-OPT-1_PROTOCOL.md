# CPU-OPT-1: stop the GPU-monitor sampler without waiting out its sleep

Status: **registered before any treatment code exists and before any final-set
CPU run.** No GPU work and no model inference. The calibration/final split is
the one pinned in `CPU-PROFILE_PROTOCOL.md` (workload sha256
`8c07eaa9e154a24398144e415c92af3f55adc55bfc6ee93615f53023d028dda4`).

## Hypothesis and basis
`CPU-TIMING-BREAKDOWN_REPORT.md` (study outcome: inconclusive due to load) left
one strong candidate: `VramMonitor.stop()` blocks about 0.5 s because the sampler
ends each pass with `time.sleep(self.interval_s)` and `stop()` joins the thread.
That study did not confirm a winner; **this experiment measures the effect
directly** instead of relying on its ranking.

**H1.** Replacing the sleep with `self.stop_event.wait(self.interval_s)` reduces
the wall time of an unchanged scripted attempt, by the sleep remainder, without
changing what the monitor records.

## The single changed variable
In `bench/telemetry.py`, `VramMonitor._sample`, line 670 only:

```diff
-            time.sleep(self.interval_s)
+            self.stop_event.wait(self.interval_s)
```

Nothing else changes: not the interval (0.5 s), the join timeout (2.0 s), the
query, the peak/gap/overhead accounting, the sample loop, any other file, any
task, the scorer or the tool. Semantics: while the event is unset, `wait` returns
after `interval_s` exactly as `sleep` did; once set, the loop condition ends the
thread at once. Sample count, peak values and `max_sample_gap_s` are therefore
unchanged for a given window; only the post-`stop()` wait disappears. This is a
reduction of waiting, not of work: no sample is skipped.

## Pinned arms (never compared against a stored number)
Both arms are worktrees of one commit `C` (the descendant of this registration
that adds the runner); the treatment arm differs from control by the diff above
and nothing else, so the tool, tasks and workload are byte-identical.

| arm | `bench/telemetry.py` sha256 | rule |
|---|---|---|
| control | `b570ab8fe11a8843c17788ce95362f9bcc9cd9b3e7cdedc7a53b75f10b01dc00` (= `v0.5`) | no change |
| treatment | recorded in the implementation PR | `git diff control treatment` is exactly the diff above: 1 file, 1 insertion, 1 deletion |

Also pinned for both arms: `bench/bench.py` `6e7ffb18…d3b5a`, `bench/classifier.py`
`a8c2b9b6…3158c`, and `experiments/cpu_workload.json`. The runner verifies all
hashes and the arm diff before measuring and refuses to run otherwise. `v0.5`
stays untouched; the treatment becomes `v0.6` only if the experiment passes and
is reviewed (a later PR bumps the version identity and frozen-source gates).

## Design
- **Final set (10 tasks), 10 repetitions, 2 arms = 200 attempts, one worker.**
  One worker keeps the critical path uncontended; concurrency is covered by the
  baseline and is not the variable.
- **Interleaved:** for every (task, repetition) pair the two arms run back to
  back, order counterbalanced (`(rep + task index)` parity picks which arm
  goes first), so drift and load hit both arms equally. Each attempt is a fresh
  subprocess measured from outside (`os.wait4`), exactly like `cpu_profile`.
- **Admission control (not a kill switch):** CPU budget 600 CPU-seconds (limit
  stays 1,200). It is checked before each pair starts; a pair already in flight
  always completes. If the projected spend would exceed the budget the runner
  starts no further pair. A run that ends short of the full matrix, or whose
  measured spend exceeds the budget, is invalid and reported as such.
- **Host state:** load average and CPU pressure recorded per pair; pairs are
  split at the median pressure into low-/high-load halves.
- **Dry run:** the runner is validated on the **calibration** tasks only, to
  test the tool. Those numbers are not part of the decision, and nothing about
  the treatment may be adjusted on any run.
- **Analysis once:** computed a single time after the full matrix; no interim
  statistics, no stopping at a good-looking point, no added repetitions.

## Correctness evidence (all required; any failure invalidates the run)
1. Every attempt passes the existing gate (exit 0, no timeout, solved, matches
   the independent reference).
2. **Identical digests:** one canonical digest per task across all repetitions
   **and both arms** (`cpu_profile.canonical_digest`).
3. **Monitor equivalence, deterministic:** a new offline test drives
   `VramMonitor` of each implementation under a **controlled clock** (the
   module's `time.time`/`time.perf_counter` replaced by one scripted fake) and a
   scripted `get_gpu_memory`, with a fixed stop time, and asserts identical
   `sample_count`, peak device/Ollama values, record keys, `monitor_overhead_s`
   and `max_sample_gap_s` for control and treatment. Timing-derived fields are
   compared only under that fake clock, never from real time. A second part
   synchronizes on the sampler having entered its interruptible wait (a barrier
   set by the scripted sampler after its query returns) and asserts the
   treatment's `stop()` returns within 50 ms while the control's waits out the
   sleep. The 50 ms bound does **not** apply while a GPU query is in flight: there
   both implementations wait for the query, bounded by the unchanged 2.0 s join
   timeout, and that case is asserted equal, not fast. A planted mutant (the
   treatment skipping a sample) must fail the test.
4. **Monitor equivalence, live:** across paired attempts the median paired
   difference in `monitor_sample_count` is 0 and no pair differs by more than 1
   (timing jitter at a sample boundary).

## Decision rule (fixed in advance)
**Estimand:** the effect of the treatment on wall time for **these 10 final
tasks** (the tasks are fixed; repetitions are the random replication). Nothing is
claimed about other tasks. The 100 pairs are not independent draws: they are 10
repetitions in each of 10 tasks, so the interval is **task-stratified**.

Primary metric: per pair, `d = wall_control - wall_treatment`. Report the
median of `d` over all 100 pairs, its 95% interval from a **task-stratified
paired bootstrap** (10,000 resamples; in each, the 10 pairs of every task are
resampled with replacement within that task, then the median of all 100 is
taken; fixed seed 20261118, stdlib `random`; percentile interval), per-task
medians, and p95. The analysis is fixed here and not altered after seeing data.

- **Minimum useful improvement:** median `d >= 0.20 s` **and** the interval's
  lower bound `> 0.10 s`. Prediction recorded now: median `d` about 0.4-0.5 s
  (the breakdown measured 0.508-0.515 s of blocking).
- **No regression:** median paired CPU difference (`cpu_treatment -
  cpu_control`) must be `<= +0.02 s` and the sampler must not add work.
- **Load robustness:** the sign of the median `d` must agree in both load halves;
  otherwise the result is **inconclusive due to load** and no gain is claimed.
- If all hold: the treatment is **adopted as the `v0.6` candidate**. The claim is
  limited to harness wall-time overhead of scripted attempts. It is **not** a
  model-quality or accuracy claim. Expected effect on a real attempt (18 s in
  SMOKE-LIVE-1) is at most about 0.5 s (under 3%), nearer 0.25 s on average;
  that is stated, not inferred from the scripted result.

## Stop conditions
1. Any attempt fails the gate, any digest differs between arms, or any pinned
   hash or the arm diff does not match: **stop**, run invalid, no gain claimed.
2. CPU budget: no new pair starts once the next pair would exceed it; a pair in
   flight completes; a short matrix or measured overspend makes the run
   **invalid**.
3. The decision rule is not met (including `d < 0.20 s`): the result is
   **negative and reported as such**. No second change, no interval or timeout
   tuning, no extra repetitions to chase significance.
4. Halves disagree: the result is **inconclusive due to load**. It is recorded
   as such and the experiment **stops**: the final set is not used a second time
   to try for a clearer answer. Any follow-up needs a fresh preregistration and
   independent evaluation data.
5. Infrastructure-failure recovery (the only exception to one run): a run that
   ends invalid for a cause outside both arms (host out-of-memory kill, full disk,
   `bwrap`/`systemd` unavailable, runner crash), decided from the failure log
   **before** any effect statistic is computed, may be rerun once in full under
   the same protocol and a new output directory. All attempts of both runs are
   preserved and disclosed. A gate failure, digest mismatch or hash mismatch is
   not an infrastructure failure; no effect statistic is ever computed on an
   invalid run.
6. A faster result obtained by skipping work never counts: samples, peak
   semantics and record fields must be unchanged (conditions 3 and 4 above).

## Out of scope
Cgroup-probe caching, sandbox launch, imports, metadata-process batching (all
below the confirmed-share bar in the breakdown); the reference-sandbox root
bind; any GPU or model run; the final set before the implementation PR is
approved; changing `v0.5`, its tag, or the EXP-2 and SMOKE-LIVE-1 records.

## Sequence
1. This protocol (docs only) is reviewed and merged.
2. Implementation PR: the runner `experiments/cpu_opt1.py`, its offline tests
   (including the equivalence test and mutants), and the treatment change, with
   calibration-set dry run only. Reviewed and merged before step 3.
3. One registered final-set run; results in `CPU-OPT-1_RESULTS.md`, raw records
   gitignored.
