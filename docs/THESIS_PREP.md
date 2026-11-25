# Thesis preparation notes (draft, documentation only)

Status: **draft for review.** Registers no experiment protocol, changes no benchmark,
runs no model or GPU work, and does not touch any future holdout. Everything below
uses existing repository evidence and published abstracts only.

## 1. Closest existing approaches and our contribution
Only abstracts and records were read; none of this is a claim about full-paper
details. Numbers are from the cited sources and must be re-checked before use.

| Theme | Closest work | Relation to this thesis |
|---|---|---|
| Small models for software-engineering agents | SWE-Protege (Kon et al. 2026), SWE-Spot (Peng et al. 2026), Devstral (Rastogi et al. 2025) | These change the **model** (post-training, specialization, a 24B open model). We hold the model fixed and change the **harness**. |
| Small models, local deployment, resource cost | SWEnergy (Tripathy et al. 2025) | Compares agent frameworks with 4B and 1.7B models on SWE-bench Verified Mini, focused on **energy**. Closest in spirit; our outcome is verified success under fixed time/token budgets on independent tasks. |
| Execution feedback | Self-Debug, Reflexion, Olausson et al. 2024 | Olausson et al. report that repair gains are modest once cost is counted and that the model's own feedback quality is the bottleneck. This **supports testing** structured, deterministic feedback (H1) rather than model-written feedback, and warns that a null result is plausible. |
| Context selection | Agentless, RepoCoder, Repoformer | Retrieval and localization are established. H2 is a deterministic, model-free selector for a small model; a localization-only ablation (as in recent RAG-repair studies) is the right comparison. |
| Training on agent trajectories | SWE-Gym | Out of scope: no training. |

**Intended contribution (no novelty claim for the ideas).** A controlled,
preregistered comparison of two harness-level interventions on an unchanged 4B local
model under fixed consumer-hardware budgets, with provenance checks and full
failed-attempt accounting. Related work uses mostly larger models or changes the
model; whether harness-level changes help a 4B model on this hardware is the gap we
test, and a null result is a valid outcome. This gap statement is based on a
limited search and must be re-checked with a systematic review.

## 2. Draft December task-audit protocol (not registered)
Purpose: produce an evaluation set independent of the 24 development tasks.
1. **Independence.** New tasks are authored or sourced separately from the 24 tasks
   and from any prompt or intervention design; no task text, test or failure log from
   the development suite informs the new set. Record the source and date of each task.
2. **Leakage prevention.** Hidden tests and reference solutions stay outside the
   model's filesystem, prompts and feedback (same boundary as the proposal). Check
   that task statements do not paste the failing test or the fix. Prefer newly
   written tasks over public snippets that may be in training data; record when a
   task derives from public material.
3. **Hidden-test quality.** Each task has a golden solution that passes and an
   unsolved repo that fails; mutants (as already done for the -v2 tasks) show tests
   reject plausible wrong fixes; tests are not over-specified to one implementation.
4. **Difficulty distribution.** Calibrate on a pilot set that is **not** the final set:
   aim for a spread with few tasks at floor (0%) or ceiling (100%), because EXP-2
   showed floor/ceiling tasks (0/3 and 3/3) carry no comparison information.
5. **Realism.** Tasks resemble ordinary maintenance work (bug fix, small feature,
   refactor) in several languages/areas; state what they do not cover.
6. **Reserve a holdout.** Split before any intervention work; the holdout is opened
   once, by a registered protocol. Nobody inspects it earlier.
7. **Outputs.** Task manifest with hashes, audit checklist per task, exclusion log.

## 3. Compute budget and power, from existing evidence only
Evidence: EXP-2 (4B, 6 tasks x 3 seeds per arm): 6/18 vs 9/18 successes, paired
discordance 4 treatment-only vs 1 control-only; mean task wall about 68 s
(1224.9 s / 18), median about 38 s, p95 about 230-280 s; about 108 s of run overhead.
EXP-2 was an order-confounded development result, not a significance claim.

Exact McNemar power (two-sided alpha 0.05), assuming discordant rates equal to the
EXP-2 observation (22% treatment-only, 6% control-only; difference 17 points) and
independent pairs:

| Paired attempts | Power at observed effect | Power at half the effect |
|---|---|---|
| 18 | 0.09 | 0.02 |
| 36 | 0.36 | 0.09 |
| 72 | 0.74 | 0.20 |
| 100 | 0.88 | 0.28 |
| 150 | 0.98 | 0.44 |

Reading: detecting an effect like EXP-2's needs on the order of 100 paired attempts
(200 attempts, about 4 h at the EXP-2 mean); detecting half of it needs well over 200
pairs. Caveats: the effect size is itself a noisy estimate from 18 pairs and is
optimistic by selection; attempts on one task are correlated, so independent-pair
power is **overstated** and the analysis should resample by task; floor/ceiling tasks
add no information, so more tasks matter more than more seeds. A proper power
analysis belongs in the baseline protocol after the task audit and baseline run.
Budgets (context, steps, wall, tokens) must be fixed identically for both arms.

## 4. Decisions needed from the professor before freezing the direction
1. Is the research question and its scope (two interventions, one 4B model, one
   consumer GPU) appropriate for a bachelor's thesis? One intervention or both?
2. What counts as sufficient evidence for the defense: a negative or null result
   with a rigorous protocol, or is a positive effect expected?
3. Required practical demonstration (tool, demo, or only the study)?
4. Acceptable preregistration level, and whether tasks may be author-written.
5. Is a second model or size needed for generality, or is one 4B model enough?
6. Expected literature review depth, reference format and language of the thesis.
7. Hardware and time limits the department accepts (one 8 GB laptop GPU, runs of
   several hours).

## 5. Not done here
No benchmark change, intervention, inference, GPU use, new protocol registration, or
holdout access. No systematic literature review; the search was a handful of queries.
