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
monitor) come from the engine's own result row.

Integrity. A run counts only if (1) the child exited 0 and did not time out,
(2) the engine reports the task solved, (3) the produced workspace, the hidden
tests installed in the scoring copy, and the engine's verification output all
match an *independent reference* built here from the pinned fixtures (the
reference solution applied to the pristine repo, the task's tests, and one
independent run of the task's test command), and (4) every deterministic field
is present. The whole invocation is valid only if the full requested matrix
completed, no run failed, and every task has one single digest across all
repetitions and all worker counts. Failed attempts keep their run directory
and log under `<out>/failed/`.

CPU budget: *admission control*, not a hard cap. Before each wave the tool
stops if the CPU already spent plus the worst per-run CPU seen so far times the
wave size would reach the budget; a wave in flight is never killed (that would
discard its evidence), so overshoot is bounded by one wave and, if it happens,
is reported as a problem. The budget covers the measured benchmark process
trees, not this tool's own hashing and the independent reference runs.

Nothing here edits the benchmark or its records. Output goes under a directory
the caller names (default `runs/cpu-profile/`, which is gitignored).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bench import bench as engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD = os.path.join(ROOT, "experiments", "cpu_workload.json")
# The registered calibration/final split (CPU-PROFILE_PROTOCOL.md). Changing the
# split means a new protocol version, so this is a pin, not a setting.
WORKLOAD_SHA256 = "8c07eaa9e154a24398144e415c92af3f55adc55bfc6ee93615f53023d028dda4"
MAX_WORKERS = 4
CPU_BUDGET_S = 1200.0  # protocol: 20 minutes of CPU time
CHILD_TIMEOUT_S = 300.0  # a hung child is killed, recorded and kept as evidence
STEPS = 6
MODEL = "qwen3.5:4b-q4_K_M"  # only names the local store entry; never contacted
TREE_SKIP = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".bench_out"}

# What a repetition must reproduce exactly (nothing timing- or path-dependent).
# The last three are measured on disk here, not taken from the engine's row.
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
    "work_sha256",
    "hidden_sha256",
    "verification_evidence_sha256",
)


def missing_fields(row: dict[str, Any]) -> list[str]:
    return [k for k in DETERMINISTIC if k not in row or row[k] is None]


def canonical_digest(row: dict[str, Any]) -> str | None:
    """None when any required field is absent: nothing is hashed as `None`."""
    if missing_fields(row):
        return None
    keep = {k: row[k] for k in DETERMINISTIC}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def tree_digest(root: str) -> str:
    """Content and layout of a tree (symlinks hashed as links, never followed)."""
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in TREE_SKIP)
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).encode()
            if os.path.islink(full):
                h.update(b"L" + rel + os.readlink(full).encode())
            else:
                with open(full, "rb") as f:
                    h.update(b"F" + rel + f.read())
    return h.hexdigest()


_TIMING = re.compile(r"\bin \d+(\.\d+)?s\b")


def evidence_digest(output: str) -> str:
    """Hash of a test run's output with timings removed: the same tests ran and
    reported the same result."""
    lines = [_TIMING.sub("in Xs", ln).strip() for ln in output.splitlines()]
    return hashlib.sha256("\n".join(ln for ln in lines if ln).encode()).hexdigest()


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
    """The registered split; refuses a file that is not the pinned one."""
    raw = pathlib.Path(WORKLOAD).read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if found != WORKLOAD_SHA256:
        raise ValueError(f"{WORKLOAD} is {found}, the protocol pins {WORKLOAD_SHA256}")
    return json.loads(raw)


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


# ------------------------------------------------------- independent reference


def staged_solution(task_id: str, dest: str) -> str:
    """The pristine repo with the reference solution applied (no engine code)."""
    tdir = os.path.join(engine.TASKS, task_id)
    work = os.path.join(dest, "work")
    shutil.copytree(os.path.join(tdir, "repo"), work)
    shutil.copytree(os.path.join(tdir, "solution"), work, dirs_exist_ok=True)
    return work


def isolated_cmd(work: str, command: str) -> list[str]:
    """The task's test command in its own bubblewrap: no network, no host home or
    environment, only `work` writable. Deliberately not `engine.bwrap_cmd`, so the
    reference does not share the code it is checking."""
    setenv = {"PATH": "/usr/bin:/bin", "HOME": "/root", "LC_ALL": "C.UTF-8"}
    cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp"]
    cmd += ["--bind", work, "/tmp/work", "--tmpfs", "/home", "--tmpfs", "/root"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--chdir", "/tmp/work"]
    cmd += ["--unshare-all", "--die-with-parent", "--clearenv"]
    for key, value in setenv.items():
        cmd += ["--setenv", key, value]
    return [*cmd, "sh", "-c", command]


def reference(task_id: str, scratch: str) -> dict[str, Any]:
    """What a correct attempt must leave behind, derived without the engine's
    scoring or sandbox: expected workspace, expected hidden tests as installed in
    the scoring copy, and the output of one independent run of the task's tests.
    The fixture is checked against the suite pin *before* any of its code runs."""
    pinned = engine.load_suite("v0.5")["tasks"][task_id]
    found = engine.task_digest(task_id)
    ref: dict[str, Any] = {"task_digest": pinned, "fixture_digest": found}
    if found != pinned:
        ref["tests_exit"] = None
        return ref
    meta = engine.load_task(task_id)
    dest = tempfile.mkdtemp(prefix=f"ref-{task_id}-", dir=scratch)
    work = staged_solution(task_id, dest)
    ref["work_sha256"] = tree_digest(work)
    hidden = os.path.join(work, "_hidden")
    shutil.copytree(os.path.join(engine.TASKS, task_id, "tests"), hidden)
    pathlib.Path(hidden, "__init__.py").touch()  # as the scorer installs them
    ref["hidden_sha256"] = tree_digest(hidden)
    try:
        done = subprocess.run(
            isolated_cmd(work, meta["test_cmd"]),
            capture_output=True,
            text=True,
            timeout=meta.get("timeout_s", 120),
            check=False,
        )
        ref["tests_exit"] = done.returncode
        ref["verification_evidence_sha256"] = evidence_digest(
            (done.stdout + done.stderr).strip()
        )
    except subprocess.TimeoutExpired, OSError:  # e.g. no bwrap: fail closed
        ref["tests_exit"] = None
        ref["verification_evidence_sha256"] = None
    shutil.rmtree(dest, ignore_errors=True)
    return ref


def reference_problems(task_id: str, ref: dict[str, Any]) -> list[str]:
    if ref["fixture_digest"] != ref["task_digest"]:
        return [f"{task_id}: the fixture on disk is not the one the suite pins"]
    if ref["tests_exit"] != 0:
        return [f"{task_id}: the independent test run did not pass on the solution"]
    return []


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
    task: str, run_dir: str, *, strict: bool, log: str, timeout_s: float
) -> tuple[dict[str, Any], int, bool]:
    """One scripted attempt in a fresh process, measured from outside."""
    cmd = [sys.executable, "-m", "experiments.cpu_profile", "one"]
    cmd += ["--task", task, "--run-dir", run_dir]
    if not strict:
        cmd.append("--no-strict-isolation")
    return measure_command(cmd, log, timeout_s)


def measure_command(
    cmd: list[str], log: str, timeout_s: float
) -> tuple[dict[str, Any], int, bool]:
    """Run `cmd` and measure it from outside (wait4 covers its descendants).
    A child past `timeout_s` is killed (its process group) and flagged; its
    directory and log are kept by the caller."""
    t0 = time.perf_counter()
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

        timer = threading.Timer(timeout_s, kill)
        timer.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        finally:
            timer.cancel()
    wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    return (
        {
            "wall_s": wall,
            "user_s": usage.ru_utime,
            "sys_s": usage.ru_stime,
            "maxrss_kib": usage.ru_maxrss,
        },
        proc.returncode,
        timed_out.is_set(),
    )


def find_attempt(run_dir: str, task: str) -> str | None:
    path = os.path.join(run_dir, "r0", task)
    return path if os.path.isdir(path) else None


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
    failed_dir: str | None = None
    refs: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    child_timeout_s: float = CHILD_TIMEOUT_S


def is_ok(row: dict[str, Any] | None) -> bool:
    """A repetition counts only if the engine says the task was solved."""
    return bool(row and row.get("passed") is True and row.get("run_successful") is True)


def disk_evidence(run_dir: str, task: str) -> dict[str, Any]:
    """Measured from the files the attempt left behind, not from the engine's row."""
    base = find_attempt(run_dir, task)
    if base is None:
        return {}
    out: dict[str, Any] = {}
    work = os.path.join(base, "work")
    hidden = os.path.join(base, "score", "_hidden")
    if os.path.isdir(work):
        out["work_sha256"] = tree_digest(work)
    if os.path.isdir(hidden):
        out["hidden_sha256"] = tree_digest(hidden)
    try:
        text = pathlib.Path(base, "test_output.txt").read_text(encoding="utf-8")
        out["verification_evidence_sha256"] = evidence_digest(text.strip())
    except OSError:
        pass
    return out


def integrity_problems(
    row: dict[str, Any] | None, evidence: dict[str, Any], ref: dict[str, Any] | None
) -> list[str]:
    if not row:
        return ["no readable results row"]
    full = {**row, **evidence}
    problems = [f"missing field {k}" for k in missing_fields(full)]
    if not is_ok(row):
        problems.append("the engine did not report the task solved")
    if ref is None:
        return [*problems, "no independent reference for this task"]
    for key in ("work_sha256", "hidden_sha256", "verification_evidence_sha256"):
        if key in evidence and evidence[key] != ref.get(key):
            problems.append(f"{key} differs from the independent reference")
    if row.get("task_digest") != ref.get("task_digest"):
        problems.append("task fixture digest differs from the suite pin")
    return problems


def record(plan: Plan, task: str, rep: int, scratch: str) -> dict[str, Any]:
    label, workers = plan.label, plan.workers
    name = f"{task}-{rep}-{workers}"
    run_dir = os.path.join(scratch, name)
    log = os.path.join(scratch, f"{name}.log")
    outer, code, timed_out = measure_child(
        task, run_dir, strict=plan.strict, log=log, timeout_s=plan.child_timeout_s
    )
    row = read_row(run_dir)
    inner = row or {}
    evidence = disk_evidence(run_dir, task)
    problems = integrity_problems(row, evidence, plan.refs.get(task))
    if code != 0:
        problems.append(f"the child exited with status {code}")
    if timed_out:
        problems.append(f"the child was killed after {plan.child_timeout_s:.0f}s")
    ok = not problems
    if not ok and plan.failed_dir:  # keep the evidence of a failed attempt
        keep = os.path.join(plan.failed_dir, f"{label}-{name}")
        shutil.rmtree(keep, ignore_errors=True)
        os.makedirs(plan.failed_dir, exist_ok=True)
        if os.path.isdir(run_dir):
            shutil.copytree(run_dir, keep, symlinks=True)
        else:
            os.makedirs(keep)
        if os.path.exists(log):
            shutil.copy(log, os.path.join(keep, "child.log"))
    shutil.rmtree(run_dir, ignore_errors=True)
    full = {**inner, **evidence}
    return {
        "set": label,
        "task": task,
        "rep": rep,
        "workers": workers,
        **outer,
        "cpu_s": outer["user_s"] + outer["sys_s"],
        "exit": code,
        "timed_out": timed_out,
        "ok": ok,
        "problems": problems,
        "commands_executed": inner.get("commands_executed"),
        "model_turns": inner.get("model_turns"),
        "engine_total_wall_s": inner.get("total_wall_s"),
        "agent_wall_s": inner.get("agent_wall_s"),
        "verification_wall_s": inner.get("verification_wall_s"),
        "monitor_overhead_s": inner.get("monitor_overhead_s"),
        "work_sha256": evidence.get("work_sha256"),
        "digest": canonical_digest(full),
    }


def run_batches(plan: Plan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """tasks x reps, `workers` at a time, under an admission-control budget (see
    the module docstring): a stop is recorded, a wave in flight is not killed."""
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
        for i in range(0, len(jobs), workers):
            wave = jobs[i : i + workers]
            spent = plan.spent_s + sum(r["cpu_s"] for r in rows)
            worst = max((r["cpu_s"] for r in rows), default=0.0)
            if spent + worst * len(wave) >= plan.budget_s:
                stopped = (
                    f"cpu budget {plan.budget_s:.0f}s: {spent:.0f}s spent, next wave "
                    f"of {len(wave)} would reach it"
                )
                break
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows += list(
                    pool.map(lambda j: record(plan, j[0], j[1], scratch), wave)
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
    """task -> its single canonical digest (the correctness reference), taken
    over *all* repetitions and worker counts; a task with two digests (or any
    run without one) has no reference."""
    seen: dict[str, set[str | None]] = {}
    for r in rows:
        seen.setdefault(r["task"], set()).add(r["digest"])
    single: dict[str, str] = {}
    for task, found in seen.items():
        only = next(iter(found))
        if len(found) == 1 and only is not None:
            single[task] = only
    return single


def validity_problems(
    rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    *,
    budget_s: float,
    reference_problem_list: list[str],
    planned_per_batch: int,
) -> list[str]:
    """Why this invocation's result must not be trusted (empty: valid).
    `planned_per_batch` is tasks x reps as the caller requested them, so an empty
    or shrunken matrix cannot agree with itself."""
    problems = list(reference_problem_list)
    if not rows:
        problems.append("no attempts were made")
    for b in batches:
        if b["planned"] != planned_per_batch:
            problems.append(
                f"workers={b['workers']}: planned {b['planned']} runs, "
                f"requested {planned_per_batch}"
            )
        if b["stopped"] or b["attempted"] != b["planned"]:
            problems.append(
                f"workers={b['workers']}: incomplete matrix "
                f"({b['attempted']}/{b['planned']}): {b['stopped']}"
            )
    for r in rows:
        if not r["ok"]:
            problems.append(
                f"{r['set']}/workers={r['workers']} {r['task']} rep{r['rep']}: "
                + "; ".join(r["problems"])
            )
    tasks = {r["task"] for r in rows}
    good = digests(rows)
    for task in sorted(tasks - set(good)):
        problems.append(f"{task}: no single deterministic digest across all runs")
    spent = sum(r["cpu_s"] for r in rows)
    if spent > budget_s:
        problems.append(f"cpu budget exceeded: {spent:.0f}s of {budget_s:.0f}s")
    return problems


def environment() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "-C", ROOT, "status", "--porcelain", "--", "bench", "experiments"],
        capture_output=True,
        text=True,
    )
    return {
        "commit": head.stdout.strip(),
        "tree_dirty": bool(dirty.stdout.strip()),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "loadavg_at_start": os.getloadavg(),
        "platform": platform.platform(),
        "note": "shared machine: other processes contribute load",
    }


# ------------------------------------------------------------------------- CLI


def render(doc: dict[str, Any]) -> str:
    lines = [f"VALID: {doc['valid']}"] if "valid" in doc else []
    lines += [f"PROBLEM: {p}" for p in doc.get("problems", [])]
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


def count_arg(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def workers_arg(text: str) -> int:
    value = count_arg(text)
    if value > MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"at most {MAX_WORKERS} concurrent workers")
    return value


def budget_arg(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and 0 < value <= CPU_BUDGET_S):
        raise argparse.ArgumentTypeError(
            f"the CPU budget must be in (0, {CPU_BUDGET_S:.0f}] seconds"
        )
    return value


def build_references(tasks: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    scratch = tempfile.mkdtemp(
        prefix="cpu-ref-", dir="/var/tmp" if os.path.isdir("/var/tmp") else None
    )
    try:
        refs = {t: reference(t, scratch) for t in tasks}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    problems = [p for t, r in refs.items() for p in reference_problems(t, r)]
    return refs, problems


def cmd_run(args: argparse.Namespace) -> int:
    try:
        doc = workload()
    except ValueError as err:
        print(f"refusing to run: {err}", file=sys.stderr)
        return 2
    tasks = doc[args.set]
    if args.set == "final" and not args.final:
        print("the final set needs --final (see the protocol)", file=sys.stderr)
        return 2
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"{out} already holds a result; choose a new --out", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    refs, ref_problems = build_references(tasks)
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for workers in args.workers if not ref_problems else []:
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
                os.path.join(out, "failed"),
                refs,
            )
        )
        rows += got
        batches.append(batch)
        if batch["stopped"]:
            break
    with open(os.path.join(out, "measurements.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    problems = validity_problems(
        rows,
        batches,
        budget_s=args.cpu_budget,
        reference_problem_list=ref_problems,
        planned_per_batch=len(tasks) * args.reps,
    )
    result: dict[str, Any] = {
        "environment": environment(),
        "workload_sha256": hashlib.sha256(
            pathlib.Path(WORKLOAD).read_bytes()
        ).hexdigest(),
        "set": args.set,
        "valid": not problems,
        "problems": problems,
        "cpu_budget_s": args.cpu_budget,
        "cpu_budget_mode": "admission-control (checked before each wave; "
        "a wave in flight is not killed, so overshoot is possible and reported)",
        "cpu_spent_s": sum(r["cpu_s"] for r in rows),
        "references": refs,
        "reference_digests": digests(rows),
        **summary(rows, batches),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(render(result))
    print(f"cpu spent {result['cpu_spent_s']:.1f}s of {args.cpu_budget:.0f}s")
    return 0 if result["valid"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    with open(os.path.join(args.dir, "summary.json"), encoding="utf-8") as f:
        doc = json.load(f)
    print(render(doc))
    return 0 if doc.get("valid") else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["one"]:
        return one_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=("calibration", "final"), required=True)
    r.add_argument("--final", action="store_true")
    r.add_argument("--reps", type=count_arg, default=3)
    r.add_argument("--workers", type=workers_arg, nargs="+", default=[1])
    r.add_argument("--out", default=os.path.join(ROOT, "runs", "cpu-profile", "run"))
    r.add_argument("--cpu-budget", type=budget_arg, default=CPU_BUDGET_S)
    r.add_argument("--no-strict-isolation", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
