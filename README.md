# AI research: small-model coding agents on an 8 GB laptop GPU

What this is: a measurement project. It asks whether a 4B-parameter model can do
real software-engineering work on an 8 GB consumer GPU faster and more reliably
than a larger, more aggressively quantized model — and how much of any gain
comes from the model versus the agent harness around it.

Hardware under test: NVIDIA RTX 5050 Laptop, 8151 MiB VRAM, 30 GB RAM, CUDA
driver 615.78.08, Ollama 0.40.13, user-level systemd unit with flash attention,
`q8_0` KV cache, single resident model.

## What is here

`bench/` — an agentic coding benchmark. 24 hand-written tasks across backend,
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

## Status

- Benchmark harness: complete, self-checked (every task fails unsolved and
  passes with its golden solution).
- Model comparison sweep: in progress. Qwen3.5-4B Q4/64K, Qwen3.5-4B Q5/64K,
  Qwen3.5-4B Q6/32K, Ornith-1.5-9B IQ3/16K.
- Retrieval / harness ablation: not started.

Results are not published yet. Do not treat any single-run number as final.

## Not a claim

This project does not reproduce weight-editing results such as Astra. Weight
editing is out of scope: the published reproductions show it repairs degenerate
reasoning loops rather than adding general capability.

## License

Apache-2.0. See `LICENSE`.
