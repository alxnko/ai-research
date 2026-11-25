# CL-FS-01: transaction-aware fault exploration, feasibility study (DRAFT)

**Status: draft, not registered, not approved.** It authorizes no tool installation,
no experiment, no product code, and no change to this repository's benchmark,
releases or experiments. It exists to be reviewed, and to be moved to its own
repository once the scope is approved. Nothing here is a result.

## Motivation
The author's own backend projects have documented defects of one class: an operation
that is interrupted or whose response is lost leaves state the client does not know
about (for example a request deadline that returns an error while the work still
commits), and retries then behave incorrectly. Ordinary endpoint tests tend to miss
these because they need a specific sequence of operations plus a fault at a specific
point. Those project records are private and are not reproduced here; the study uses
public synthetic fixtures only.

## Question
Can existing tools detect and reproduce this class of defect within a bounded budget,
and does a clear, practically important gap remain that a narrower method could fill?

- **RQ1.** Do existing tools find the known defects within a fixed execution budget?
- **RQ2.** Do they reproduce failures reliably and shrink them to short sequences?
- **RQ3.** Does a reproducible gap remain in detection, fault-window coverage,
  reproducibility or integration effort?

## Prior work that bounds any claim
No novelty is claimed for combining stateful testing, fault injection and automatic
discovery. To be read before the study (citations to be checked and completed):
- Filibuster, systematic service-level fault injection (Meiklejohn et al.).
- Assad, Meiklejohn, Miller, Krusche (2024), *Can My Microservice Tolerate an
  Unreliable Database? Resilience Testing with Fault Injection and Visualization*,
  ICSE 2024 Companion (demonstration paper), arXiv:2404.01886. It extends Filibuster
  with database-client fault injection (including PostgreSQL) for JVM services.
- Jepsen's analyses of ambiguous transaction outcomes and unsafe retries.
- Tools: fast-check (model-based testing, shrinking, asynchronous scheduler),
  Schemathesis (stateful OpenAPI testing), Toxiproxy (network faults), Antithesis
  (deterministic simulation, commercial).

## Fault model (initial, deliberately narrow)
A **failure window** is a point in one operation where a fault changes the relation
between what the client believes, what the application executed and what the database
durably committed. Controlled application-level windows only:
1. interrupted before commit (no prohibited partial effects);
2. client deadline expires while processing continues (recovery follows the declared
   operation contract);
3. commit succeeds but the response is lost, and the client retries (no duplicate
   logical effect where idempotency is promised).

Also: duplicate requests and recovery operations. Out of scope: operating-system
determinism, arbitrary database interleavings, distributed infrastructure failures,
and the database's internal commit protocol. A seed reproduces the generated operations
and the fault schedule; it does not promise identical thread or database interleavings.

## Fixture and defect corpus (public, synthetic)
A small TypeScript HTTP service on PostgreSQL with an injectable clock and explicit
test-only fault points, in three modules: authentication challenges, payments, and
reservations. Each operation has a written contract and executable invariants.

Every defect has a corrected twin as a negative control.

**Core corpus** (inside the stated fault model; decides the study):

| ID | Defect | Class | Set |
|---|---|---|---|
| D01 | deadline expires but the operation still commits (challenge consumed, result lost) | timeout recovery | development |
| D02 | response lost after payment commit; retry duplicates the effect | idempotency | development |
| D03 | partial update survives a failed operation | atomicity | development |
| H01 | expired payment cannot be retried as specified | recovery / state transition | evaluation |
| H05 | idempotency key reused across distinct operations | deduplication | evaluation |

**Exploratory corpus** (outside the initial fault model; reported separately, never
used to decide, so the study does not drift into a general application-testing
platform):

| ID | Defect | Class | Set |
|---|---|---|---|
| H02 | wrong behavior around a business-day cutoff | clock | exploratory |
| H03 | concurrent requests create conflicting reservations | concurrency | exploratory |
| H04 | out-of-order confirmation overwrites a terminal state | event ordering | exploratory |

Every defect has a corrected twin as a negative control.

The evaluation and exploratory cases are written and frozen before any baseline runs. Because the author knows
these categories, H-cases are not unseen; **a stronger evaluation uses additional
defects authored independently** (supervisor or another developer, not the tool
author) and kept hidden until the method is frozen. Leakage prevention: hidden defects
are delivered as a sealed, hashed bundle outside the repository and CI, are never run
against any tool before the freeze, and their author sees no tool output before it. Ground truth comes from independent regression tests and
manual inspection, not from the tools under test.

## Methods compared
- **B0** conventional integration tests with the same invariants.
- **B1** fast-check model-based testing with its scheduler.
- **B2** fast-check plus the same fault instrumentation, operation model, recovery
  operations and invariants the proposed method would get. **B2 is the key baseline:**
  without it, instrumentation gains could be credited to a new exploration policy.
- **B3** Schemathesis stateful testing with the needed custom checks.
- **P1** a proposed exploration policy, built only if B0-B3 leave a meaningful gap.
- Filibuster is studied as a related implementation (JVM-oriented); cross-language
  comparison limits are recorded, and it is not forced into the numeric comparison.

Equal information, fault access, invariants, wall-clock and execution budgets across
methods; wall-clock cost is reported separately.

## Measurements
- Defect detection rate (core corpus; exploratory reported separately); false
  positives on corrected twins; time and executions to first detection; replay
  success rate over repeated replays; minimized sequence length; peak RAM and CPU.
- **Fault-window coverage** = distinct (operation, window-type) pairs on the declared
  list of reachable windows that were exercised at least once, divided by the list size.
  The list is written with the fixture, before any run.
- **Reproducibility** = replays of a recorded (sequence, fault schedule) that reproduce
  the same invariant violation, out of a fixed number of replays.
- **Integration effort** = lines of integration code, hours (logged), configuration
  items, required application modifications, and external dependencies.
- **Capability coverage** (separate from algorithmic results): for each tool, which
  cases and fault types it supports at all. UNSUPPORTED and BLOCKED outcomes stay
  visible there and are never counted as algorithmic MISSES.
- Outcomes per case: DETECTED, MISSED, FALSE POSITIVE, UNSUPPORTED, INCONCLUSIVE,
  BLOCKED.
- **Two budget views, reported separately:** fixed execution count, and fixed wall
  clock. Neither is used alone.
- Preserve tool versions, configurations, revisions, fault schedules, seeds,
  environment and raw output.

## Freeze points
1. **Before any baseline runs:** fixtures, corpora, invariants, window list, feasibility
   budgets (execution count, wall clock, seeds, replay count) and the decision
   criteria below.
2. **Before P1 is evaluated (only if built):** the formal evaluation budgets, seeds and
   sample sizes, on **separate evaluation data** not used in the feasibility run.
Thresholds are not tuned after seeing results; deviations are recorded.

## Controls and integrity
- Pin runtimes, dependencies and container images; isolated Linux environment.
- Replays run independently; distinguish reproducible schedules from deterministic
  execution.
- Record deviations; never change criteria after seeing results.
- Disposable databases and synthetic data only. No private repositories, production
  data, real payment or authentication services, credentials, or paid model inference.
- Run unfamiliar tools in isolated environments with minimal permissions.

## Decision criteria (fixed at freeze point 1; proposed values, core corpus only)
- **ADEQUATE (adopt existing tools):** B2 detects every core evaluation defect within
  the feasibility budget, with no false positive on the twins, replays each failure in
  at least 9 of 10 replays, and shrinks each failure to at most twice the length of a
  hand-minimized sequence.
- **GAP (a specialized method may be justified):** at least one core defect is missed
  by every baseline B0-B3 under both budget views, or replay or shrinking criteria
  fail for every baseline, and the cause is a method limitation, not missing
  instrumentation (checked by B2) and not an UNSUPPORTED or BLOCKED outcome.
- **INDETERMINATE:** anything else, including BLOCKED or INCONCLUSIVE results that
  prevent a clear call. Fix the cause (not the criteria) and rerun once, recorded.

Decision rule:
1. ADEQUATE: recommend adopting existing tools, or a narrowly justified upstream
   contribution.
2. GAP and practically important: propose a specialized implementation and register a
   formal evaluation protocol before building it.
3. No defensible contribution or manageable scope: keep the original research thesis
   or another supervisor-approved candidate.

Whatever the outcome, the comparison itself is reportable.

## Scope notes
- Graduation scope if approved: HTTP APIs, PostgreSQL, stateful operation generation,
  controlled fault injection, invariant checking, shrinking and replay, TypeScript
  integration. Other stacks (Python, Java, Dart), WebSocket testing, MCP integration
  and any dashboard are future work. A claim that the method transfers across
  applications needs its own study and is not part of this one.
- Candidates are not ranked here. The professor's decision on direction precedes any
  work.

## Staging
Feasibility (this document) -> graduation scope (core engine, below) -> adoption in
other projects -> longer-term product. Later stages are possibilities, not
commitments; the feasibility result decides whether the earlier ones are justified.
The design should keep adapters separable so other stacks remain possible later.

## Open items before registration
- Verify and complete the prior-work citations, including Filibuster's own papers.
- Choose exact tool versions and the fixture's fault-point mechanism.
- Confirm the window list, replay count and the proposed decision thresholds.
- Decide who authors the independent hidden defects.
- Supervisor approval of direction and scope.
