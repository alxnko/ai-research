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

| ID | Defect | Class | Set |
|---|---|---|---|
| D01 | deadline expires but the operation still commits (challenge consumed, result lost) | timeout recovery | development |
| D02 | response lost after payment commit; retry duplicates the effect | idempotency | development |
| D03 | partial update survives a failed operation | atomicity | development |
| H01 | expired payment cannot be retried as specified | state transitions | evaluation |
| H02 | wrong behavior around a business-day cutoff | time | evaluation |
| H03 | concurrent requests create conflicting reservations | concurrency | evaluation |
| H04 | out-of-order confirmation overwrites a terminal state | ordering | evaluation |
| H05 | idempotency key reused across distinct operations | deduplication | evaluation |

The H-set is written and frozen before any baseline runs. Because the author knows
these categories, H-cases are not unseen; **a stronger evaluation uses additional
defects authored independently** (supervisor or another developer) and kept hidden
until the method is frozen. Ground truth comes from independent regression tests and
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
Defect detection rate; false positives (on corrected twins); fault-window coverage;
time and executions to first detection; replay success rate; minimized sequence
length; integration effort (lines of integration code and hours, logged); peak RAM and
CPU. Outcome per case: DETECTED, MISSED, FALSE POSITIVE, UNSUPPORTED, INCONCLUSIVE,
BLOCKED. Unsupported and blocked cases are never counted as algorithmic failures.
Preserve tool versions, configurations, revisions, fault schedules, seeds, environment
and raw output. No numeric success targets are set now; budgets, seeds, sample sizes
and thresholds are frozen after the baseline measurements and before P1 is evaluated.

## Controls and integrity
- Pin runtimes, dependencies and container images; isolated Linux environment.
- Replays run independently; distinguish reproducible schedules from deterministic
  execution.
- Record deviations; never change criteria after seeing results.
- Disposable databases and synthetic data only. No private repositories, production
  data, real payment or authentication services, credentials, or paid model inference.
- Run unfamiliar tools in isolated environments with minimal permissions.

## Decision rule (before any run)
1. Existing tools adequately cover the scenarios: recommend adopting them, or a
   narrowly justified upstream contribution.
2. A reproducible, practically important gap remains: propose a specialized
   implementation and register a formal evaluation protocol before building it.
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

## Open items before registration
- Verify and complete the prior-work citations, including Filibuster's own papers.
- Choose exact tool versions and the fixture's fault-point mechanism.
- Define "fault-window coverage" and "integration effort" precisely.
- Decide who authors the independent hidden defects.
- Supervisor approval of direction and scope.
