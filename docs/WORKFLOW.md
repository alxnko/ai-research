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

## Experiment controller

`experiments/controller.py` derives the whole attempt matrix from one manifest
(`experiments/manifests/`) and enforces it; offline, no Ollama or GPU needed.

- `check` (manifest vs suite pins), `analyze`, `plan`, and `run`. `analyze`
  validates every recorded attempt against the manifest (fixture digest, model
  identity, prompt hash, settings, seed, exact bench invocation), binds the
  record files to their cell by hash (`attempt.json` for new runs; a frozen
  manifest names a committed `SHA256SUMS` via `integrity`, so even the original
  `runs/exp2` is hash-checked), and reports malformed records as
  invalid instead of crashing.
- `analyze` exits non-zero while any attempt is missing, invalid or unexpected
  and withholds the decision. A zero-success arm is `no_success` (0.0 tasks/h).
  If control has no successes the ratio is undefined and the rule is not
  evaluated (a deliberate, conservative reading; the protocol is silent).
  The decision uses exact arithmetic on unrounded wall time; rounding is display-only.
- `run` takes an exclusive lock and reserves each attempt (`<cell>.partial`)
  before launching, so even a launcher that never starts leaves evidence; it
  stages each attempt in that directory,
  validates it, then promotes it with one rename. Invalid or interrupted
  attempts stay as `.invalid-N` / `.partial`, and the next `run` refuses until a
  person files them under `<runs>/aborted/` (ignored by the matrix, kept as
  evidence), so a failed attempt is never silently re-rolled. Only the manifest's
  arm directories are policed; `aborted/`, `logs/` etc. beside them are not.
  An arm may name its own `launch` entry (which must write the provenance
  file, as `exp2_entry` does); otherwise the generic launcher is used. Plain
  `bench.py` is never launched directly.
- `experiments/manifests/exp2.json` describes frozen EXP-2 (analysis only).
  `experiments/exp2_records/` holds byte copies of its 36 `config.json`,
  `exp2_policy.json` and `results.json` (hashes in `SHA256SUMS`; `env.json` and
  transcripts stay in the ignored `runs/`). Tests reproduce 6/18 vs 9/18 and
  1.51x from them; with `runs/exp2` (or `EXP2_RUNS`) present they also check the
  copies match the originals, otherwise that check reports SKIPPED.
- **Code identity (run-only).** A runnable manifest registers
  `benchmark.tag`, the full `commit`, `source_files` (engine, classifier,
  telemetry, package init) and `source_sha256` (digest over those files plus the
  suite manifest); `python -m experiments.source_identity TAG SUITE FILES...`
  computes them from the tag's objects. `verify` / `run` fail closed unless the
  files on disk hash to that digest, each selected task matches its suite pin,
  no source file is a symlink or missing, the tag resolves to the registered
  commit, and each file and task is byte-identical to the object at that
  commit. The checkout's HEAD is irrelevant (newer controller code may run a
  frozen benchmark). `check`, `plan` and `analyze` stay offline and never need
  the tag. Needs the tag: `git fetch --tags`.
- **Generic launcher** (`experiments/launcher.py`) starts any arm without its
  own `launch`. Before a model call it re-checks the bench arguments, the code
  identity, that the loaded engine modules are the verified files, the model's
  manifest/config/weights digests, and that the arm's prompt (engine base prompt
  plus the manifest's `prompt_edit`, applied exactly once) hashes to the pinned
  `prompt_sha256` and `system_prompt_sha256`. A failure leaves
  `launch-refused.json` and exits 4 with the engine not started. Then it runs
  the unchanged engine behind a recorder that hashes the system message of every
  request that reaches the model call; only if every request carried the pinned
  prompt (and at least one was made) does it write `provenance.json`. Otherwise
  `launch-failed.json` is written and the attempt is quarantined.
- Limits: `run` has not been exercised against a live model; the decision rule
  compares only inference errors and timeouts, as the protocol words it.

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
