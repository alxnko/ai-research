# Agentic coding benchmark

Measures whether a local model can complete real engineering work on a fixed
VRAM budget — not whether it wins a vendor benchmark.

## What it measures

Per task, `bench.py` runs a shell-only agent loop (one `RUN: <command>` per
turn, mini-SWE-agent style) against a disposable git repo, then scores the
result with hidden tests the agent never sees.

Recorded per run: pass/fail, steps, tokens, wall time, peak VRAM, tool calls,
invalid actions, changed files, unexpected changes.

Derived in the summary: task completion rate, **verified tasks per hour**,
regression rate (files touched outside `allowed_files`).

## Tasks

24 hand-written tasks in `tasks/`, each with:

- `repo/` — the starting repository, stub or buggy code
- `tests/` — hidden tests, copied in only at scoring time as `_hidden/`
- `solution/` — golden solution, used by `selfcheck.py`
- `meta.json` — category, prompt, test command, timeouts, allowed files

Categories: backend, debugging, devops, frontend, mobile, refactoring.

Tasks are independent, offline, and use only the Python standard library.
No network, no package installs.

Python 3.15.2 (uv-managed) is bound read-only into the sandbox as
`/opt/python`, because the sandbox masks `/home`. Override the interpreter
with `BENCH_PYTHON_HOME` if the uv path changes.

## Sandbox

`bwrap` with a read-only root, tmpfs `/home` and `/root`, a private `/tmp`, the
worktree bound at `/mnt/work`, and `--unshare-net`. The agent cannot reach the
host filesystem, the network, or anything outside its worktree.

## Usage

```bash
uv run bench.py --list
python3 bench.py --model qwen3.5:4b --ctx 65536 --ngpu 99 --steps 20 --tag q4-64k
python3 bench.py --tasks backend-pagination debug-tz-offset --steps 15
python3 selfcheck.py          # task + parser regression check
```

`--ngpu 99` forces full GPU offload. It is a per-profile flag, never a global
default: under GPU pressure a forced offload can fail to allocate.

Results land in `runs/<stamp>-<tag>/` with `results.json`, `env.json`
(versions, driver, power limit, model digests, Ollama unit environment) and one
`transcript.json` per task.

## Adding a task

1. Create `tasks/<id>/{repo,tests,solution}` and `meta.json`.
2. Make `repo/` fail the hidden tests and `solution/` pass them.
3. Run `python3 selfcheck.py` — it must print `ok` for the new task.

`selfcheck.py` also regression-tests `parse_action`, the agent's only actuator:
multi-line commands (heredocs) must survive parsing intact.

## Known limits

- One run per task by default; `--repeat N` is available but expensive.
- VRAM sampling is coarse (0.5 s) and the desktop baseline drifts.
- Bash-only devops tasks report absent tooling as a task failure; check the
  transcript before blaming the model.
