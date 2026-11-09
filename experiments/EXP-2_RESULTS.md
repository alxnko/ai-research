# EXP-2 results: generic verification policy (Qwen3.5-4B)

**Status: COMPLETE and frozen.** 36/36 scheduled attempts produced a valid
record. Reported before any change to the runner, prompt, tasks or analysis.
Protocol: `experiments/EXP-2_PROTOCOL.md`. Core: tag `v0.4` (`da36111`).
Runner: `experiments/run_exp2.sh`. Analysis: `experiments/analyze_exp2.py`.

Executed 2026-11-09, control arm first, then treatment, strictly sequential.

## Headline

| Arm | Successes | Rate (95% Wilson) | Verified tasks/hour | Total task wall |
| --- | --- | --- | --- | --- |
| control | 6/18 | 33.3% (16–56%) | 17.63 | 1224.9 s |
| verification | 9/18 | 50.0% (29–71%) | 26.54 | 1220.6 s |

Throughput ratio **1.51x**, raw successes **+3**, paired discordance
**4 treatment-only / 1 control-only**. The preregistered decision rule is met:
>=1.25x primary throughput, at least as many raw successes, no new inference
errors or deadline timeouts. Treat it as a **provisional operational** result,
not a significance claim: the intervals overlap and N=18 per arm.

Gain came from correctness, not speed: total task wall differs by 4.3 s
(0.4%). The treatment completed three more tasks per end-to-end hour at the
same cost.

## Per task (successes of 3 seeds)

| Task | control | verification |
| --- | --- | --- |
| `backend-auth-token` | 0/3 | 2/3 |
| `backend-idempotency` | 0/3 | 0/3 |
| `debugging-off-by-one` | 3/3 | 3/3 |
| `debugging-money-round` | 0/3 | 0/3 |
| `refactor-parser` | 3/3 | 2/3 |
| `refactor-validators` | 0/3 | 2/3 |

Discordant cells: treatment-only wins on auth-token (seed 42, 44) and
validators (seeds 43, 44); control-only win on parser (seed 44). No seed
flipped a solved easy task in a way that suggests noise alone: the ceiling
tasks stayed at 3/3 and the floor tasks stayed at 0/3 in both arms.

## Outcomes

| Arm | passed | hidden_test_failure | invalid_action |
| --- | --- | --- | --- |
| control | 6 | 10 | 2 |
| verification | 9 | 9 | 0 |

Reliability: 0 inference errors, 0 deadline timeouts, 0 zero-token runs,
0 incomplete token records, 0 forbidden-file changes, 0 added or removed
files outside the allowed set in either arm.

Caveat: the `invalid_action` outcome disappeared in the treatment (2 runs to
0), but total invalid action events rose from 2 to 5. Every affected run
recovered within the turn budget. The protocol's decision rule counts
inference errors and timeouts as reliability regressions; under a stricter
reading of "no new reliability regression", that event increase is a real, if
small, cost of the longer verification instruction.

## Other measurements

| Metric | control | verification |
| --- | --- | --- |
| Median task wall | 37.8 s | 34.1 s |
| p95 task wall | 230.0 s | 280.4 s |
| Aggregate decode | 67.6 tok/s | 71.1 tok/s |
| Prompt tokens (all runs) | 843,606 | 734,181 |
| Eval tokens (all runs) | 78,610 | 82,800 |
| Max prompt tokens | 16,213 | 15,604 |
| Peak VRAM | 6,110 MiB | 6,348 MiB |

The treatment wrote 13% fewer prompt tokens while producing 5% more
evaluation tokens: fewer turns spent re-reading, more turns spent writing and
checking. Decode rate difference is likely thermal/order drift (control ran
first), not a property of the prompt.

Observed window: first attempt started 14:33:58, last finished 15:16:31
(2,553 s). Sum of task walls is 2,445 s, so process start-up and model
set-up contributed 108 s outside the per-task metric.

## Two aborted attempts (excluded)

- `control seed42 backend-auth-token`: launching shell was killed after ~90 s.
- `control seed42 backend-idempotency`: sweep stopped when the runner gained
  result validation.

Both are preserved under `runs/exp2/aborted/` with a note, contain no
`results.json`, and were rerun. They are not counted as model failures.

## Audit

- 36/36 arm x seed x task cells exist exactly once, task field matches path.
- One model fingerprint across all 36 runs:
  manifest `sha256:d8b0f5e9…e38c04`, config `sha256:788a4d94…2f84db`,
  weights `sha256:a9d0a3e1…f7e2ac50`.
- One prompt SHA per arm: control `e3e28dd7…f073`, treatment `27a9d510…8d37`.
- Every run: `native_tools`, `strict_isolation=true`, `ctx=16384`,
  `ngpu=99`, `steps=20`, `max_wall=600`, sampling
  `exp2-coding-v1` (temp 0.6, top-p 0.95, top-k 20, repeat-penalty 1.0, seed
  as registered), tag `v0.4`.
- `ollama ps` reports `context_length: 16384`; the service's
  `OLLAMA_CONTEXT_LENGTH=65536` does not apply to these requests.

## Limitations

- N=18 per arm, 6 tasks x 3 seeds; intervals overlap. No significance claim.
- Fixed order (all control, then all treatment) confounds arm with time,
  thermals and cache state, as the protocol already states.
- Tasks were selected after EXP-1 experience, so this is a development suite,
  not an unbiased sample of coding ability.
- Environment drifted since EXP-1C: Ollama 0.40.15 (was 0.40.13), driver
  620.70.06 (was 615.78.08). Both arms share it; historical v0.2 numbers are
  not directly comparable.
- The 18 holdout v0.3/v0.4 tasks were untouched and remain available for
  validation.

## What this says about the bottleneck

`backend-idempotency` (0/6) and `debugging-money-round` (0/6) failed in both
arms, so this prompt does not address them. Control's two idempotency runs
ended on invalid tool syntax with the response truncated at the length limit;
treatment's idempotency runs reached the hidden tests and still failed. That
is a capability/horizon limit, not an interaction-format limit, and it is the
right place to look next rather than tuning the verification wording further.

Next steps, in order: (1) validate the policy on the 18 holdout tasks with the
same runner; (2) analyse the two floor tasks for horizon versus reasoning
failure; (3) only then move to context/retrieval and quantization work.
