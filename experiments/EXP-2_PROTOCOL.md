# EXP-2 preregistration: generic verification policy (Qwen3.5-4B)

**Status:** PRE-REGISTERED BEFORE MEASUREMENT. Do not change this protocol, task set, policy text, or inference/runtime controls after the first EXP-2 result without creating a new version and separate treatment identifier.

## Rationale and falsifiable hypothesis

EXP-1C (historical v0.2) reported Qwen3.5-4B Q4_K_M native-tool performance of 8/18 successes in 1,228.1 seconds (23.45 verified tasks/hour) on six deliberately selected tasks and seeds 42, 43, 44. That is a **historical reference**, not the primary EXP-2 baseline.

**H1:** Replacing the baseline agent's generic "Ensure all changes are verified before finishing" instruction with a general verification and repair instruction yields >=25% improvement in **verified successful attempts per end-to-end hour** at matched model, seed, task, sampling, budget, sandbox and controller, without reducing total verified successes.

**H0 / failure to demonstrate H1:** the improvement threshold is not met, overall correctness regresses, or the purported improvement relies on invalid/incomplete runs. Treat small-sample observations as exploratory even when H1's operational threshold is met.

## Frozen controller and identical arm settings

Benchmark core: git annotated tag **v0.4**, target commit `da36111`; controller fixes must be identical for both arms. The EXP-2-only wrapper `experiments/exp2_entry.py` changes **only** the system prompt selected before invoking `bench.bench.main()`. The wrapper and this protocol must be committed **before** starting either arm; retain that final commit SHA and its file hashes in the analysis.

- Model: **`qwen3.5:4b-q4_K_M`**, with weights/config digests and full manifest fingerprint matching historical EXP-1C. The `qwen3.5:4b` smoke-test tag is *not* an acceptable substitute unless exact digest equivalence is verified and recorded.
- Actuator: `native_tools`; Ollama `/api/chat`, existing `execute_command` and `finish_task` tool schemas.
- `--ctx 16384 --ngpu 99 --steps 20 --max-wall 600 --strict-isolation`.
- `--temperature 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.0`, seeds 42, 43, 44. No model thinking toggle or implicit changes to `num_predict`.
- Same Ollama version/settings (`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`), GPU driver and environment; record actual values and monitor background GPU VRAM and other workload changes. Benchmark v0.3 records some but not necessarily all of these settings, so capture supplemental service metadata.
- Strict bubblewrap/cgroup v2. No hidden evaluator tests, solutions, or post-hoc failure hints passed to agent. Public tests and the agent's own tests are permitted.
- Unique ignored run directories: `runs/exp2/{control|verification}/seed{42|43|44}/<task>/`. Never overwrite prior attempts. `exp2_policy.json` records the exact prompt SHA and invocation.

## Independent variable: system-prompt sentence only

**Control:** the unmodified `SYSTEM_PROMPT_TOOLS` from v0.3.

**Treatment:** same system prompt with exactly one substitution. Replace:

`Ensure all changes are verified before finishing.`

with the following fixed text, verbatim:

> Before finishing, verify the requested behavior using available public tests and targeted checks for relevant boundary conditions, input types, error handling, and behavioral invariants. If a check fails, inspect the failure and repair the implementation. Do not access hidden evaluation tests.

All other system-prompt lines, user task instructions, tools, and budgets are unchanged. The generic policy must not contain task-specific references to idempotency, concurrency, monetary rounding, or the private tests.

## Core version and model identity (recorded before the first attempt)

The protocol originally froze v0.3 (`dd3f6bc`). Before any EXP-2 attempt,
the core was advanced to **v0.4** (`da36111`) because v0.3 could not satisfy
preflight: `model_fingerprint()` read only the legacy
`~/.ollama/models/manifests` layout, so Ollama 0.40 tags written under
`manifests-v2/` were recorded as the 12-char `ollama list` ID with no weights
digest, and the CI lint gate had been failing since `2086ef3`. v0.4 changes
only manifest resolution, code formatting and lint configuration. Sandbox,
scoring, agent loop, prompts and telemetry semantics are byte-identical.
Both arms run on v0.4; no EXP-2 result exists for any earlier core.

Verified identity for the registered model, and for the `qwen3.5:4b` tag used
in the v0.3 smoke test:

| tag | manifest digest | config digest | weights digest |
| --- | --- | --- | --- |
| `qwen3.5:4b-q4_K_M` | `sha256:d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` | `sha256:788a4d94bc3f431ac0cd067005be4b838a09330427962952420b05d10ecf84db` | `sha256:a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50` |
| `qwen3.5:4b` | `sha256:d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` | `sha256:788a4d94bc3f431ac0cd067005be4b838a09330427962952420b05d10ecf84db` | `sha256:a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50` |

The two tags resolve to the same manifest, config and weights layer; they
differ only in baked-in sampling defaults, which the runner always overrides.
The wrapper still requires the registered tag and records the fingerprint in
`exp2_policy.json`.

## Preselected tasks (six, all in both arms)

1. `backend-auth-token`
2. `backend-idempotency`
3. `debugging-off-by-one`
4. `debugging-money-round`
5. `refactor-parser`
6. `refactor-validators`

Run seeds **42, 43, 44** for each arm: 18 control + 18 treatment = **36 attempts**. This is the development/diagnostic suite, deliberately chosen after EXP-1 experience and therefore not an unbiased benchmark sample.

**Execution order:** perform all 18 control attempts first, then all 18 treatment attempts, in fixed seed order (42,43,44), and within each seed in the task order above. Run strictly sequentially; explicitly record this order and interpret results as potentially confounded by thermal conditions, time-of-day, service cache effects, and temporal drift. Do not interleave new interventions halfway through the registered run.

**Holdout:** the 18 other v0.3 benchmark tasks are reserved for follow-up validation and may not be tuned against during this phase. Any follow-up validation uses the same matched policy comparison; report it as validation on public synthetic fixtures, not independent real-world SWE performance.

## Outcome and analysis plan

**Primary system metric:** `verified_tasks_per_hour = 3600 * sum(run_successful) / sum(total_wall_s)` with failures, timeouts, inference errors and verification included for every scheduled task. Use both arms' fresh v0.3 results. Historical v0.2 throughput is reported separately.

**Secondary metrics:** raw successful counts and 6x3 paired pass/fail matrix; task-level discordances; inference errors and deadline timeouts; zero-token and partial-stream events; invalid tool actions; command/turn counts; total prompt/eval tokens (mark interrupted token accounting as incomplete), actual model-generated output where available; median/p95 elapsed task time, token-weighted decode throughput, GPU memory.

**Scientific reporting:** include descriptive uncertainty (e.g. Wilson intervals for each proportion and task/seed paired discordances); do not claim significance or generality from N=18 per arm. Aggregate success/time comparisons should show the numerator and denominator and not exclude failed or censored runs. Separately report coding-evaluable samples, but never replace raw denominator by infrastructure-adjusted denominator.

**Decision rule:** call the treatment a provisional operational winner only if >=1.25x primary throughput AND at least as many raw successes as control, with no new reliability/safety regressions. Otherwise report no clear win and choose the next intervention based on observed failures. Do not cherry-pick tasks or seeds. Any stability, controller, policy or task changes require a newly identified follow-up experiment.

## Preflight and stop rules

1. Verify current code against v0.4 core (diff `bench/bench.py`, `bench/telemetry.py`, `bench/selfcheck.py` exactly zero vs tag); wrapper/protocol files are the only experiment additions.
2. Verify target Qwen alias manifest, weights/config digest, GPU layers and Ollama service settings; ensure the exact model is tested, not the separate `qwen3.5:4b` tag used in the v0.3 smoke. The wrapper fails closed if the resolved fingerprint differs from the registered one.
3. Run `uv run python bench/selfcheck.py` and a small **non-scored** prompt-wrapper parity check that compares the two fully expanded system prompts. Do not use scored tasks as an informal pre-run selection procedure.
4. If missing results, model mismatch, broken isolation, controller crash, GPU offload mismatch or unrecoverable Ollama fault occurs, pause experiment for infrastructure investigation. Coding failures and deadline outcomes are valid and do not stop the study.
5. Preserve every run, including infra failures, and mark any rerun/replacement explicitly. Report all attempted experiments and any deviations; never silently omit/censor errors.

## Running the frozen experiment wrapper

Invoke from the repository root. Example for one control task:

```bash
uv run python -m experiments.exp2_entry --exp2-policy control \
  --run-dir runs/exp2/control/seed42/backend-auth-token \
  --model qwen3.5:4b-q4_K_M --actuator native_tools --task backend-auth-token \
  --ctx 16384 --ngpu 99 --steps 20 --max-wall 600 --strict-isolation \
  --sampling-profile exp2-coding-v1 --temperature 0.6 --top-p 0.95 \
  --top-k 20 --repeat-penalty 1.0 --seed 42
```

Treatment uses `--exp2-policy verification` and `runs/exp2/verification/...`; all other flags identical. A separate sequencer may launch each one-task invocation, but it must not modify or bypass the frozen wrapper, skip invalid results, or overwrite completed output.

**Provenance note:** the cancellation hardware checks and `selfcheck.py` pass are reported by the local operator, not independently executed by reviewers without laptop access.
