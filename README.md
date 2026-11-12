# AI research: small-model coding agents on an 8 GB laptop GPU

What this is: a measurement project. It asks whether a 4B-parameter model can do
real software-engineering work on an 8 GB consumer GPU faster and more reliably
than a larger, more aggressively quantized model — and how much of any gain
comes from the model versus the agent harness around it.

Research device: NVIDIA RTX 5050 Laptop with 8,151 MiB VRAM. Driver,
Ollama version, and runtime settings differ by experiment; each run's
`env.json` and registered protocol are the source of truth for provenance.

## Quick start for contributors and coding agents

```bash
make help        # discover commands; no GPU jobs start
make doctor      # check local prerequisites
make setup       # sync the locked Python environment
make check       # Ruff + format + ty + offline sandbox/golden selfchecks
make report      # read EXP-2 results without generating anything
```

Repository agent instructions are in [`AGENTS.md`](AGENTS.md). Claude Code
uses [`CLAUDE.md`](CLAUDE.md) to import the same rules, with no duplicate
policy to maintain. For task-specific procedures read
[`docs/WORKFLOW.md`](docs/WORKFLOW.md). GPU experiments are explicit,
versioned operations and are **not** part of ordinary verification.

## What is here

`bench/` — an agentic coding benchmark. 24 hand-written tasks (suite `v0.4`; corrected suite `v0.5`) across backend,
debugging, devops, frontend, mobile and refactoring. Each task is a disposable
repo with hidden tests, a golden solution, and a shell-only agent loop that
runs under `bwrap` with no network and no access to the host home directory.

Scored on task completion, verified tasks per hour, regression rate, tool-call
success, tokens, and peak VRAM. Every run records the Ollama version, driver,
model digest and server settings alongside the numbers.

See `bench/README.md` for usage and task authoring.

## Benchmark versions

**v0 (commit `3e9d40b`) — pilot.** The first 24-task suite. Known defects, fixed
in v0.1 but kept frozen so pilot numbers stay comparable:

- the task set is Python standard-library logic in all six categories; it is
  cross-domain programming logic, not native frontend, Android or React work;
- the scorer copied hidden tests into an agent-writable directory, so a
  symlinked `_hidden` could redirect writes onto the host;
- `passed` ignored forbidden file changes, and regression detection missed
  untracked files;
- no per-task wall-clock, memory or output cap;
- the sandbox exposed the whole host filesystem read-only, not just the
  toolchain it needs.

**v0.1–v0.4 — hardened runner.**

- `v0.1` curates the sandbox mounts, rebuilds a clean scoring workspace from
  regular files only, makes forbidden changes fail scoring, caps output and
  process resources, and snapshots the workspace instead of trusting `.git`.
- `v0.2` adds streaming telemetry, two-axis failure classification, the
  native-tools actuator, and cgroup v2 isolation.
- `v0.3` moves inference to a `forkserver` worker with an absolute monotonic
  deadline, cancels in-flight requests at the deadline, and marks interrupted
  token accounting as incomplete.
- `v0.4` reads Ollama's v2 manifest layout, so each run records the full
  manifest, config and weights digests, and restores green lint, format and
  type gates.

Tags `v0.2`–`v0.4` are annotated and pushed. Experiments pin a tag; see
`experiments/EXP-2_PROTOCOL.md` for the first preregistered comparison.

## Status

- Benchmark harness: complete, self-checked (every task fails unsolved and
  passes with its golden solution); CI runs the self-check under bubblewrap.
- EXP-2 (verification prompt policy, Qwen3.5-4B): 6/18 control vs 9/18
  treatment, 17.63 vs 26.54 verified tasks/hour at equal total task time.
  Provisional: N=18 per arm, overlapping intervals. See
  `experiments/EXP-2_RESULTS.md`.
- Retrieval / harness ablation: not started.

Results are not published yet. Do not treat any single-run number as final.

## Not a claim

This project does not reproduce weight-editing results such as Astra. Weight
editing is out of scope: the published reproductions show it repairs degenerate
reasoning loops rather than adding general capability.

## License

Apache-2.0. See `LICENSE`.
