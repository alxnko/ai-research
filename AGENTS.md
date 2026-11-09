# AI research contributor guide

This repository is a measurement system, **not** a general coding-agent demo.
Keep experiments reproducible, comparison conditions explicit, and generated
claims proportional to the evidence. Prefer a small correct diff over new
frameworks or abstractions.

## Start here

- `make help` lists the developer commands. `make setup` synchronizes `uv.lock`.
- `make check` is the canonical local CI-equivalent verification (`./check.sh`).
- `make selfcheck` runs offline harness, sandbox, and 24 golden/baseline checks.
- `make report` reads EXP-2 results; it does not launch the GPU experiment.
- See `bench/README.md` for benchmark internals, `experiments/EXP-2_PROTOCOL.md`
  for the registered experiment, and `docs/WORKFLOW.md` for task-specific steps.

## Two different agent roles

- **Engineering AI (you):** maintain code, tests, documentation, and experiments
  using ordinary repository development tools. You may inspect scoring fixtures
  only when auditing test/scorer quality, never to tailor a model's evaluated
  responses, prompts, or patches to hidden answers.
- **Model under evaluation:** is a separate, untrusted, restricted process that
  sees only its task checkout and public task instructions. Never inject this
  file, hidden tests, golden solutions, or research analysis into its context.

## Non-negotiable boundaries

- Do not change or retag released benchmark versions. Do not modify active
  experiment model settings, prompts, task fixtures, scorer, or run records.
  If a flaw is found, record it and prepare a new, versioned experiment.
- Keep `bench/tasks/*/tests/` and `solution/` out of the evaluated model's
  filesystem, tool output, and prompt. Preserve scorer-owned clean copies.
- Never commit credentials, personal paths, host logs, or raw run transcripts.
  Treat the public GitHub repository as non-secret; `runs/` stays gitignored.
- Don't make live model/GPU runs a default CI, commit-hook, or formatting action.
  They require an explicit, preflighted experiment protocol and GPU budget.
- Preserve strict sandbox isolation; never silently fall back to unsandboxed
  shell execution or privileged local access for evaluated model commands.

## Work efficiently

- Read only the relevant code and task guidance. Don't scan every fixture or
  rerun a full GPU sweep for a small change. Use existing tools before adding
  dependencies; format and static-check changes deterministically, without AI.
- For code edits, add a regression test for changed behavior and run `make check`.
  For docs-only changes, run `git diff --check` and verify examples/links.
- Record what changed, what was actually tested, and any unverified assumption.
  A passing selfcheck is not proof of research validity or sandbox perfection.
- New experiments need a frozen protocol, identified model/harness/prompt/task
  hashes, explicit stop rules, complete failed-attempt accounting, and a holdout.
