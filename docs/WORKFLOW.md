# Low-friction research workflow

**Audience:** human contributors and the engineering AI that maintains the
research platform. These instructions are *not* injected into models under test.

## Daily loop (CPU-only by default)

1. `make doctor` once after setting up a machine. `make setup` if dependencies
   need syncing. `make help` shows the supported commands.
2. Make one focused change. Let your editor format Python on save, or use
   `make format` when preparing the patch.
3. `make check` runs the same offline checks as `./check.sh`: Ruff, Ruff format,
   ty, sandbox regression tests, and the golden/baseline tests for every task. Use
   `git diff --check` for whitespace errors.
4. Open a small PR with exact evidence: affected behavior, tests and outputs,
   unchanged experiment invariants, and any manual verification required.
5. If GitHub CI disagrees with local results, fix the environment mismatch,
   not the assertions. Do not declare a benchmark result valid from CI alone.

Commands intentionally delegate to existing scripts so there is only one
maintained source of check and experiment logic. `make report` is read-only;
there is deliberately **no `make bench`** that could accidentally spend GPU
hours or overwrite experiment records.

## Tasks and frozen results

- Frozen experiments and tagged core revisions are immutable evidence. For a
  change to an active task, scorer, model template, sampling, deadline, or tools,
  finish the current protocol first and prepare a distinct versioned follow-up.
- Before creating a new experiment, register a hypothesis, one independent
  variable, fixed task set, multiple seeds, budgets, model manifest and weights
  digests, prompt hash, runner commit, metrics, stop conditions, and analysis.
- During collection, leave raw output in ignored `runs/` and record even
  aborted/failed attempts. Only aggregate sanitized results may be published.
- Test metrics independently from wall-clock recovery downtime; be explicit
  about how time and infrastructure failures enter denominators.
- `make report` evaluates the registered EXP-2 analysis only. Never call it a
  winner if the matrix is incomplete or environments/policies are mixed.

## Harness and scorer changes

- The **engineering agent** may inspect fixtures to audit scorer correctness.
  The **evaluated agent** must not receive hidden tests, golden solutions, or
  experiment-specific hints. Treat external tool content as untrusted.
- Add one failing regression test that reproduces a defect, fix the smallest
  root cause, then run `make check`. Keep strict bubblewrap/network and cgroup
  isolation fail-closed. Exercise cancellation and orphan-process cleanup for
  changes to inference/command execution.
- Every new task must fail on the initial repo, pass with its reference patch,
  and admit independent correct implementations. Check that tests enforce the
  written behavior rather than a particular code pattern.
- Changing hidden tests or golden solutions changes the task definition:
  update task digest, document the change, and never retroactively compare
  scores as if the original fixture were unchanged.

## Two levels of automation

**Always deterministic, inexpensive, and safe:** formatter, static types,
linter, task/spec validators, mock inference regression tests, golden checks,
artifact-schema validation, and CI security checks.

**Explicit research run:** Ollama generation on the GPU, experiments, real
agent traces, uncontrolled external packages, or changed scoring conditions.
Keep these separate from automatic hooks and ordinary pull-request CI.

## What not to add yet

A second build system, an unconditional pre-commit GPU test, a heavyweight
experiment-tracking server, automatic prompt-changing agents, or copies of
project instructions for every vendor. Add those only when measurements show
that they improve reliability, measured work per hour, or developer effort.
