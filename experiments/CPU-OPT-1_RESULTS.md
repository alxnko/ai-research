# CPU-OPT-1 results: interruptible monitor wait

Protocol: `CPU-OPT-1_PROTOCOL.md` (registered in PR #20 at `2669938`, amended
at `77b80d5` before any implementation, and by Amendment 1 in PR #21 before any
final-set run). Registered decision: **ADOPT AS v0.6 CANDIDATE**. This is the
only final-set run; nothing was rerun, tuned or recovered.

## Identity
| item | value |
|---|---|
| code commit (run from) | `8b60df995f3f6177321c1d6004543a84ec64144f` (merge of PR #21), clean tree |
| protocol merge / amendments | `e807c09` (PR #20: `2669938`, `77b80d5`); Amendment 1 in `9d5658c` |
| treatment | `experiments/cpu_opt1_treatment.patch` sha256 `22f6caf0...63ed`, applied to a worktree of the same commit |
| `bench/telemetry.py` sha256 | control `b570ab8f...dc00` (= `v0.5`), treatment `0b8b001b...cae1f` |
| other pinned files | `bench/bench.py` `6e7ffb18...d3b5a`, `bench/classifier.py` `a8c2b9b6...3158c`, `cpu_workload.json` `8c07eaa9...8dda4`: identical in both arms |
| command | `python3 -m experiments.cpu_opt1 run --set final --registered-final --out runs/cpu-opt1/final-v1` |
| host | shared Linux machine, 12 CPUs, Python 3.14.10; load average at start 5.0 / 5.3 / 6.7 |
| scope | 10 registered final tasks, 10 repetitions, 2 arms, one worker, strict isolation; no GPU work, no model |

The change under test is one line in `VramMonitor._sample`:
`time.sleep(self.interval_s)` -> `self.stop_event.wait(self.interval_s)`.

## Gates (all passed)
- 200 of 200 attempts passed the engine gate and the independent reference; every
  cell of the registered (task, repetition, arm) matrix occurs exactly once.
- One canonical digest per task across all repetitions and both arms (10 digests).
- Monitor equivalence, deterministic and real-thread (preflight): passed.
  Live, every attempt recorded exactly one monitor sample in both arms; that
  check is weak on these short attempts, the deterministic test carries the
  equivalence evidence.
- CPU spent 52.1 s of the 600 s budget. Frozen `bench` sources unchanged
  (`bench.py`, `classifier.py`, `telemetry.py`, `__init__.py` identical to `v0.5`
  on `main`).

## Result (100 pairs, 10 tasks x 10 repetitions)
| | value | registered requirement |
|---|---|---|
| median paired wall reduction `d` | **0.495 s** | >= 0.20 s |
| 95% interval, task-stratified paired bootstrap | [0.4925, 0.5009] s | lower bound > 0.10 s |
| median CPU difference (treatment - control) | +0.001 s | <= +0.02 s |
| arm medians, wall | control 0.916 s, treatment 0.418 s (about 54% lower) | n/a |
| p95 of `d` | 0.549 s | n/a |
| pairs with `d > 0` | 100 of 100 (range 0.375 to 0.667 s) | n/a |
| per-task median `d` | 0.489 to 0.508 s, all 10 tasks | n/a |
| load groups (median pressure cut 2.92) | low 49 pairs, high 49 pairs (2 at the cut) | >= 20 each |
| median `d` by group | low +0.4946 s, high +0.4962 s | same sign |

Host CPU pressure (`avg10`) ranged 0.86 to 11.4 across the pairs. Load policy as
registered; no window was chosen and load was not manipulated.

## Independent recomputation
The summary's statistics were recomputed from the raw `attempts.jsonl` with the
runner's own `validity_problems` and `analyse` after the run: no validity
problems, statistics identical to the stored summary, decision identical.

## What this supports, and what it does not
Supported: replacing the monitor's unconditional sleep with an interruptible
event wait reduced median paired harness wall time by 0.495 s on the ten
registered scripted tasks on this machine, with the registered correctness and
monitoring-equivalence checks intact. The interval describes repeated
measurements of these 10 fixed tasks, not other tasks or machines.

Not supported: any model-quality or accuracy claim; extending the 54% to real
agent runs. Those take far longer (SMOKE-LIVE-1: 18 s), where the same 0.5 s is at
most about 3% and about 0.25 s on average. The live sample-count check had one
sample per attempt. The reservation guard protects this checkout only.

## Evidence manifest (raw records are local and gitignored)
| file | sha256 |
|---|---|
| `runs/cpu-opt1/final-run.json` (reservation) | `fb7939d132ba221cbbddf8bb2e78c72009b9d9ff2d8e8685f7470028ff0f4879` |
| `runs/cpu-opt1/final-v1/attempts.jsonl` (200 rows) | `70c664201fb3c1a37689119c4128a404226aa5aa6a07f95e5b79ef7f77be9545` |
| `runs/cpu-opt1/final-v1/summary.json` | `d00740ceee0b58e154431d32639e5916c724f294edc7329e42c6699f71b0887c` |
| `runs/cpu-opt1/final-v1/started.json` | `dbae3777304e11e3d27ce16e150ba44abfc02d831a2ef8601ce3cdbdc4adc79d` |

Preserve these files; do not delete or recreate them to retry. The calibration
dry runs (`dryrun-v1`, `dryrun-v2`) were tool validation only and are not part of
the decision.

## Next
Adopting the change as `v0.6` is a separate reviewed release PR (apply exactly the
validated patch, update the version identity and frozen-source gates, keep `v0.5`
immutable, tag only after merge). It is not done here.
