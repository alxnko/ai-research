"""CPU-side cost of the benchmark's own machinery, with no model inference.

    python -m experiments.cpu_profile run --set calibration --reps 3 --out DIR
    python -m experiments.cpu_profile report DIR

Protocol: experiments/CPU-PROFILE_PROTOCOL.md. A scripted actor stands in for
the model: for each pinned task it writes the task's reference solution through
the engine's own tool-call path (so command execution, the sandbox, the
workspace, the GPU monitor and the hidden-test verification all run exactly as
for a real attempt), then calls `finish_task`. It is a load generator, not a
model: no score from it says anything about a model.

Every repetition is a fresh process running the unchanged engine's `main()`
(the way the controller starts an attempt), measured from outside with
`wait4`: wall time, user and system CPU of the whole process tree and the peak
resident set of its largest process. Per-phase times (agent, verification,
monitor) come from the engine's own result row. Each run also yields a canonical
digest of its deterministic outputs; repetitions and later (optimised) versions
must agree on it, so "faster" can never mean "did less".

Nothing here edits the benchmark or its records. Output goes under a directory
the caller names (default `runs/cpu-profile/`, which is gitignored).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bench import bench as engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD = os.path.join(ROOT, "experiments", "cpu_workload.json")
MAX_WORKERS = 4
CPU_BUDGET_S = 1200.0  # protocol: 20 minutes of CPU time
STEPS = 6
MODEL = "qwen3.5:4b-q4_K_M"  # only names the local store entry; never contacted

# What a repetition must reproduce exactly (nothing timing- or path-dependent).
DETERMINISTIC = (
    "task",
    "task_digest",
    "passed",
    "run_successful",
    "outcome",
    "verification_passed",
    "changed_files",
    "added_files",
    "removed_files",
    "forbidden_files",
    "commands_executed",
    "model_turns",
    "tool_calls",
    "invalid_actions",
    "tokens",
)


def canonical_digest(row: dict[str, Any]) -> str:
    keep = {k: row.get(k) for k in DETERMINISTIC}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": nearest_rank(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def workload() -> dict[str, Any]:
    with open(WORKLOAD, encoding="utf-8") as f:
        return json.load(f)


def workload_problems(doc: dict[str, Any], suite: dict[str, Any]) -> list[str]:
    cal, fin = set(doc["calibration"]), set(doc["final"])
    problems = []
    if cal & fin:
        problems.append(f"calibration and final overlap: {sorted(cal & fin)}")
    if cal | fin != set(suite["tasks"]):
        problems.append("calibration + final must be exactly the suite's tasks")
    if len(cal) != len(doc["calibration"]) or len(fin) != len(doc["final"]):
        problems.append("duplicate task in a set")
    return problems


# --------------------------------------------------------------- scripted actor


def solution_command(task_dir: str) -> str:
    """One shell command that writes the reference solution into /mnt/work."""
    root = os.path.join(task_dir, "solution")
    parts = []
    for dirpath, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            with open(full, "rb") as f:
                blob = base64.b64encode(f.read()).decode()
            parent = os.path.dirname(rel)
            mk = f"mkdir -p '{parent}' && " if parent else ""
            parts.append(f"{mk}printf %s '{blob}' | base64 -d > '{rel}'")
    return " && ".join(parts) or "true"


def scripted(task_id: str):
    """Replacement for the engine's mock model: same telemetry shape, but the
    first turn applies the task's reference solution."""
    original = engine._mock_chat_stream
    command = solution_command(os.path.join(engine.TASKS, task_id))

    def chat(model, msgs, ctx, ngpu, *args, **kwargs):
        result = original(model, msgs, ctx, ngpu, *args, **kwargs)
        if kwargs.get("step", 0) == 0:
            call = {
                "id": "script_1",
                "function": {
                    "name": "execute_command",
                    "arguments": {"command": command},
                },
            }
            return dataclasses.replace(result, tool_calls=[call])
        return result

    return chat


def one_main(argv: list[str]) -> int:
    """Child: one attempt through the unchanged engine, scripted."""
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--no-strict-isolation", action="store_true")
    args = p.parse_args(argv)
    os.environ["BENCH_MOCK_MODEL"] = "1"
    engine._mock_chat_stream = scripted(args.task)
    cmd = ["--model", MODEL, "--actuator", "native_tools", "--task", args.task]
    cmd += ["--ctx", "8192", "--steps", str(STEPS), "--max-wall", "180"]
    cmd += ["--run-dir", args.run_dir]
    if not args.no_strict_isolation:
        cmd.append("--strict-isolation")
    sys.argv = ["bench.py", *cmd]
    return int(engine.main() or 0)


# ------------------------------------------------------------------- measuring


def measure_child(
    task: str, run_dir: str, *, strict: bool
) -> tuple[dict[str, Any], int]:
    """Run one child and measure it from outside (wait4 covers its descendants)."""
    cmd = [sys.executable, "-m", "experiments.cpu_profile", "one"]
    cmd += ["--task", task, "--run-dir", run_dir]
    if not strict:
        cmd.append("--no-strict-isolation")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    return {
        "wall_s": wall,
        "user_s": usage.ru_utime,
        "sys_s": usage.ru_stime,
        "maxrss_kib": usage.ru_maxrss,
    }, proc.returncode


def read_row(run_dir: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
            rows = json.load(f)
        return rows[0] if isinstance(rows, list) and len(rows) == 1 else None
    except OSError, ValueError:
        return None


@dataclasses.dataclass(frozen=True)
class Plan:
    label: str
    tasks: list[str]
    reps: int
    workers: int
    strict: bool
    budget_s: float
    spent_s: float = 0.0


def is_ok(row: dict[str, Any] | None) -> bool:
    """A repetition counts only if the engine says the task was solved."""
    return bool(row and row.get("passed") is True and row.get("run_successful") is True)


def record(plan: Plan, task: str, rep: int, scratch: str) -> dict[str, Any]:
    label, workers = plan.label, plan.workers
    run_dir = os.path.join(scratch, f"{task}-{rep}-{workers}")
    outer, code = measure_child(task, run_dir, strict=plan.strict)
    row = read_row(run_dir)
    shutil.rmtree(run_dir, ignore_errors=True)
    inner = row or {}
    ok = is_ok(row)
    return {
        "set": label,
        "task": task,
        "rep": rep,
        "workers": workers,
        **outer,
        "cpu_s": outer["user_s"] + outer["sys_s"],
        "exit": code,
        "ok": ok,
        "commands_executed": inner.get("commands_executed"),
        "model_turns": inner.get("model_turns"),
        "engine_total_wall_s": inner.get("total_wall_s"),
        "agent_wall_s": inner.get("agent_wall_s"),
        "verification_wall_s": inner.get("verification_wall_s"),
        "monitor_overhead_s": inner.get("monitor_overhead_s"),
        "digest": canonical_digest(inner) if row else None,
    }


def run_batches(plan: Plan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """tasks x reps, `workers` at a time. A budget stop is recorded, never hidden:
    the CPU spent so far is checked before each run starts."""
    workers = max(1, min(plan.workers, MAX_WORKERS))
    plan = dataclasses.replace(plan, workers=workers)
    jobs = [(task, rep) for rep in range(plan.reps) for task in plan.tasks]
    rows: list[dict[str, Any]] = []
    stopped = None
    scratch = tempfile.mkdtemp(
        prefix="cpu-profile-", dir="/var/tmp" if os.path.isdir("/var/tmp") else None
    )
    t0 = time.perf_counter()
    try:
        # chunk by `workers` so the budget is re-checked between waves
        for i in range(0, len(jobs), workers):
            if plan.spent_s + sum(r["cpu_s"] for r in rows) >= plan.budget_s:
                stopped = f"cpu budget {plan.budget_s:.0f}s reached"
                break
            wave = jobs[i : i + workers]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows += list(
                    pool.map(
                        lambda j: record(plan, j[0], j[1], scratch),
                        wave,
                    )
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    wall = time.perf_counter() - t0
    done = sum(1 for r in rows if r["ok"])
    batch = {
        "set": plan.label,
        "workers": workers,
        "attempted": len(rows),
        "planned": len(jobs),
        "stopped": stopped,
        "wall_s": wall,
        "workflows_per_hour": done / wall * 3600 if wall else 0.0,
    }
    return rows, batch


def summary(rows: Iterable[dict[str, Any]], batches: list[dict[str, Any]]):
    rows = list(rows)
    out: dict[str, Any] = {"batches": batches, "groups": {}}
    keys = ("wall_s", "cpu_s", "user_s", "sys_s", "maxrss_kib")
    phases = (
        "agent_wall_s",
        "verification_wall_s",
        "monitor_overhead_s",
        "engine_total_wall_s",
    )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["set"], r["workers"]), []).append(r)
    for (label, workers), group in sorted(groups.items()):
        entry: dict[str, Any] = {"runs": len(group)}
        entry["execution_failures"] = sum(1 for r in group if not r["ok"])
        for k in (*keys, *phases):
            vals = [r[k] for r in group if isinstance(r.get(k), int | float)]
            if vals:
                entry[k] = summarize(vals)
        walls = [r["wall_s"] for r in group]
        eng = [r["engine_total_wall_s"] for r in group if r["engine_total_wall_s"]]
        if eng and len(eng) == len(group):  # process start-up and teardown
            entry["outside_engine_s"] = summarize(
                [w - e for w, e in zip(walls, eng, strict=True)]
            )
        by_task: dict[str, set[str | None]] = {}
        for r in group:
            by_task.setdefault(r["task"], set()).add(r["digest"])
        entry["nondeterministic_tasks"] = sorted(
            t for t, d in by_task.items() if len(d) != 1 or None in d
        )
        out["groups"][f"{label}/workers={workers}"] = entry
    return out


def digests(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """task -> its single canonical digest (the correctness reference)."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        if r["digest"]:
            seen.setdefault(r["task"], set()).add(r["digest"])
    return {t: sorted(d)[0] for t, d in seen.items() if len(d) == 1}


def environment() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "commit": head.stdout.strip(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "loadavg_at_start": os.getloadavg(),
        "platform": platform.platform(),
        "note": "shared machine: other processes contribute load",
    }


# ------------------------------------------------------------------------- CLI


def render(doc: dict[str, Any]) -> str:
    lines = []
    for name, g in doc["groups"].items():
        lines.append(f"{name}: {g['runs']} runs, {g['execution_failures']} failures")
        for key in (
            "wall_s",
            "cpu_s",
            "agent_wall_s",
            "verification_wall_s",
            "monitor_overhead_s",
            "outside_engine_s",
            "maxrss_kib",
        ):
            if key in g:
                s = g[key]
                lines.append(
                    f"  {key:22s} median {s['median']:9.3f}  p95 {s['p95']:9.3f}"
                    f"  min {s['min']:9.3f}  max {s['max']:9.3f}"
                )
        if g["nondeterministic_tasks"]:
            lines.append(f"  NONDETERMINISTIC: {g['nondeterministic_tasks']}")
    for b in doc["batches"]:
        stop = f" STOPPED: {b['stopped']}" if b["stopped"] else ""
        lines.append(
            f"batch {b['set']} workers={b['workers']}: {b['attempted']}/"
            f"{b['planned']} runs in {b['wall_s']:.1f}s = "
            f"{b['workflows_per_hour']:.0f} workflows/hour{stop}"
        )
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    doc = workload()
    tasks = doc[args.set]
    if args.set == "final" and not args.final:
        print("the final set needs --final (see the protocol)", file=sys.stderr)
        return 2
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"{out} already holds a result; choose a new --out", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for workers in args.workers:
        spent = sum(r["cpu_s"] for r in rows)
        got, batch = run_batches(
            Plan(
                args.set,
                tasks,
                args.reps,
                workers,
                not args.no_strict_isolation,
                args.cpu_budget,
                spent,
            )
        )
        rows += got
        batches.append(batch)
        if batch["stopped"]:
            break
    with open(os.path.join(out, "measurements.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    result: dict[str, Any] = {
        "environment": environment(),
        "workload_sha256": hashlib.sha256(
            pathlib.Path(WORKLOAD).read_bytes()
        ).hexdigest(),
        "set": args.set,
        "cpu_spent_s": sum(r["cpu_s"] for r in rows),
        "reference_digests": digests(rows),
        **summary(rows, batches),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(render(result))
    print(f"cpu spent {result['cpu_spent_s']:.1f}s of {args.cpu_budget:.0f}s")
    failed = any(g["execution_failures"] for g in result["groups"].values())
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    with open(os.path.join(args.dir, "summary.json"), encoding="utf-8") as f:
        print(render(json.load(f)))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["one"]:
        return one_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=("calibration", "final"), required=True)
    r.add_argument("--final", action="store_true")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--workers", type=int, nargs="+", default=[1])
    r.add_argument("--out", default=os.path.join(ROOT, "runs", "cpu-profile", "run"))
    r.add_argument("--cpu-budget", type=float, default=CPU_BUDGET_S)
    r.add_argument("--no-strict-isolation", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
