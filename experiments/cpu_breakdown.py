"""Read-only timing breakdown of one scripted attempt (no inference, no GPU use).

    python -m experiments.cpu_breakdown run --reps 5 --out DIR
    python -m experiments.cpu_breakdown micro --reps 30 --out DIR
    python -m experiments.cpu_breakdown report DIR

Plan and reporting rules: experiments/CPU-TIMING-BREAKDOWN.md (registered before
any measurement). The frozen engine is not edited: the traced child replaces
`subprocess.Popen` and wraps engine functions in its own process only, the same
way the scripted actor in `cpu_profile` replaces the mock model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from bench import bench as engine
from bench import telemetry
from experiments import cpu_profile as cp

EVENTS: list[dict[str, Any]] = []
MAIN = "MainThread"
SPANS = {  # engine function -> span name
    "main": "engine_main",
    "prepare": "prepare",
    "run_sandboxed": "sandbox",
    "check_cgroup_v2": "cgroup_probe",
    "_kill_cgroup_scope": "cgroup_kill",
    "assert_sandbox_python": "assert_python",
    "git_snapshot": "git_snapshot",
}
COMPONENTS = (
    "startup",
    "imports",
    "engine_other",
    "prepare",
    "git_snapshot",
    "assert_python",
    "cgroup_probe",
    "sandbox_run",
    "cgroup_kill",
    "monitor_start",
    "monitor_stop",
    "teardown",
)
# the isolated measurement that must reproduce each component's per-call cost
MICRO_FOR = {
    "cgroup_probe": "probe",
    "sandbox_run": "scope_bwrap_true",
    "cgroup_kill": "kill_absent",
    "startup": "py_bare",
    "monitor_stop": "monitor_stop",
}
CPU_BUDGET_S = 300.0


# ------------------------------------------------------------- traced child

_CODE = (
    "import sys,time\nT=time.time()\n"
    "from experiments import cpu_breakdown as b\n"
    "sys.argv[1]=repr(T)\nsys.exit(b.child())"
)


def _tag(args: Any) -> str:
    if isinstance(args, str):  # a shell line: keep enough to tell them apart
        return "sh:" + args.split("|")[0].strip()[:40]
    argv = [str(a) for a in args]
    head = os.path.basename(argv[0])
    if head == "systemd-run":
        return "systemd-run:probe" if argv[-1] == "true" else "systemd-run:scope"
    return head


class TimedPopen(subprocess.Popen):
    """Records spawn cost (fork/exec) and lifetime of every child process."""

    def __init__(self, args: Any, *a: Any, **kw: Any) -> None:
        t0 = time.monotonic()
        super().__init__(args, *a, **kw)
        self._ev: dict[str, Any] = {"kind": "proc", "tag": _tag(args)}
        self._ev |= {"thread": _thread(), "t0": t0, "spawn_s": time.monotonic() - t0}
        EVENTS.append(self._ev)

    def wait(self, timeout: float | None = None) -> int:
        try:
            return super().wait(timeout)
        finally:
            if self.returncode is not None:
                self._ev.setdefault("t1", time.monotonic())


def _thread() -> str:
    return threading.current_thread().name


def span(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*a: Any, **kw: Any) -> Any:
        t0, c0 = time.monotonic(), time.thread_time()
        try:
            return fn(*a, **kw)
        finally:
            EVENTS.append(
                {"kind": "span", "name": name, "thread": _thread(), "t0": t0}
                | {"t1": time.monotonic(), "cpu_s": time.thread_time() - c0}
            )

    return wrapper


def install() -> None:
    """Patch this process only; the engine and telemetry files are untouched."""
    subprocess.Popen = TimedPopen  # ty: ignore[invalid-assignment]
    for attr, name in SPANS.items():
        setattr(engine, attr, span(name, getattr(engine, attr)))
    mon = telemetry.VramMonitor
    setattr(mon, "start", span("monitor_start", mon.start))  # noqa: B010
    setattr(mon, "stop", span("monitor_stop", mon.stop))  # noqa: B010
    sample = span("monitor_sample", telemetry.get_gpu_memory)
    setattr(telemetry, "get_gpu_memory", sample)  # noqa: B010


def child() -> int:
    """Entry of a traced attempt. argv: T spawn task run_dir trace strict."""
    t_entry = time.time()
    t_first, spawn, task, run_dir, trace, strict = sys.argv[1:7]
    install()
    code = cp.one_main(
        ["--task", task, "--run-dir", run_dir]
        + ([] if strict == "1" else ["--no-strict-isolation"])
    )
    doc = {
        "spawn": float(spawn),
        "t_first": float(t_first),
        "t_entry": t_entry,
        "t_exit": time.time(),
        "pynvml": telemetry.HAS_PYNVML,
        "events": EVENTS,
    }
    with open(trace, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return code


# ------------------------------------------------------------------ analysis


def exclusive_times(spans: list[dict[str, Any]]) -> dict[str, float]:
    """Per span name: time not covered by a directly nested span."""

    def dur(x: dict[str, Any]) -> float:
        return x["t1"] - x["t0"]

    def holds(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
        return (
            outer is not inner
            and outer["t0"] <= inner["t0"] <= inner["t1"] <= outer["t1"]
        )

    own = {id(s): dur(s) for s in spans}
    for c in spans:
        parents = [p for p in spans if holds(p, c)]
        if parents:  # the tightest enclosing span owns c
            own[id(min(parents, key=dur))] -= dur(c)
    total: dict[str, float] = {}
    for s in spans:
        total[s["name"]] = total.get(s["name"], 0.0) + own[id(s)]
    return total


def breakdown(trace: dict[str, Any], outer: dict[str, Any]) -> dict[str, Any]:
    events = trace["events"]
    main = [e for e in events if e["kind"] == "span" and e["thread"] == MAIN]
    excl = exclusive_times(main)
    calls = [e for e in main if e["name"] == "sandbox"]
    comp = {
        "startup": trace["t_first"] - trace["spawn"],
        "imports": trace["t_entry"] - trace["t_first"],
        "engine_other": excl.get("engine_main", 0.0),
        "sandbox_run": excl.get("sandbox", 0.0),
        "teardown": outer["wall_s"] - (trace["t_exit"] - trace["spawn"]),
    }
    for name in COMPONENTS:
        if name in excl:
            comp[name] = excl[name]
    comp = {k: comp.get(k, 0.0) for k in COMPONENTS}
    procs = [e for e in events if e["kind"] == "proc" and "t1" in e]
    samples = [e for e in events if e.get("name") == "monitor_sample"]
    tags: dict[str, dict[str, float]] = {}
    for p in procs:
        t = tags.setdefault(p["tag"], {"n": 0, "spawn_s": 0.0, "life_s": 0.0})
        t["n"] += 1
        t["spawn_s"] += p["spawn_s"]
        t["life_s"] += p["t1"] - p["t0"]
    return {
        "components": comp,
        "critical_path_s": sum(comp.values()),
        "sandbox_calls": len(calls),
        "per_call": {
            "cgroup_probe": excl.get("cgroup_probe", 0.0) / max(1, len(calls)),
            "sandbox_run": excl.get("sandbox", 0.0) / max(1, len(calls)),
            "cgroup_kill": excl.get("cgroup_kill", 0.0) / max(1, len(calls)),
        },
        "procs": tags,
        "monitor": {
            "backend": "pynvml" if trace["pynvml"] else "nvidia-smi",
            "samples": len(samples),
            "wall_s": sum(s["t1"] - s["t0"] for s in samples),
            "cpu_s": sum(s["cpu_s"] for s in samples),
        },
    }


def pressure() -> float | None:
    try:
        with open("/proc/pressure/cpu", encoding="ascii") as f:
            return float(f.readline().split("avg10=")[1].split()[0])
    except OSError, IndexError, ValueError:
        return None


# -------------------------------------------------------------- attempt runner


def attempt(task: str, kind: str, refs: dict[str, Any], scratch: str, strict: bool):
    name = f"{task}-{kind}-{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(scratch, name)
    log = os.path.join(scratch, f"{name}.log")
    trace = os.path.join(scratch, f"{name}.trace.json")
    host = {"load1": os.getloadavg()[0], "cpu_pressure10": pressure()}
    spawn = time.time()
    command = None
    if kind == "traced":  # argv[1] is replaced by the child with its own T
        command = [sys.executable, "-c", _CODE, "0", repr(spawn), task, run_dir, trace]
        command.append("1" if strict else "0")
    if command is None:
        outer, exit_code, timed_out = cp.measure_child(
            task, run_dir, strict=strict, log=log, timeout_s=cp.CHILD_TIMEOUT_S
        )
    else:
        outer, exit_code, timed_out = cp.measure_command(
            command, log, cp.CHILD_TIMEOUT_S
        )
    row = cp.read_row(run_dir)
    evidence = cp.disk_evidence(run_dir, task)
    problems = cp.integrity_problems(row, evidence, refs.get(task))
    if exit_code != 0 or timed_out:
        problems.append(f"child exit {exit_code}, timed_out={timed_out}")
    out = {"task": task, "kind": kind, **outer, **host, "ok": not problems}
    out |= {"problems": problems, "cpu_s": outer["user_s"] + outer["sys_s"]}
    out["digest"] = cp.canonical_digest({**(row or {}), **evidence})
    if kind == "traced" and os.path.exists(trace):
        with open(trace, encoding="utf-8") as f:
            tr = json.load(f)
        out["breakdown"] = breakdown(tr, outer)
        out["trace"] = tr
    if problems:  # keep a failed attempt's evidence next to the raw records
        out["failed_log"] = log
    return out


# ------------------------------------------------------------------------ run


def cmd_run(args: argparse.Namespace) -> int:
    doc = cp.workload()
    tasks = doc["calibration"]
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"{out} already holds a result; choose a new --out", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    refs, ref_problems = cp.build_references(tasks)
    rows: list[dict[str, Any]] = []
    stopped = None
    scratch = tempfile.mkdtemp(prefix="cpu-breakdown-", dir="/var/tmp")
    try:
        order = [(r, t, i) for r in range(args.reps) for i, t in enumerate(tasks)]
        for rep, task, i in order if not ref_problems else []:
            kinds = ["traced", "plain"] if (rep + i) % 2 == 0 else ["plain", "traced"]
            spent = sum(r["cpu_s"] for r in rows)
            worst = max((r["cpu_s"] for r in rows), default=0.0)
            if spent + 2 * worst >= args.cpu_budget:
                stopped = f"cpu budget {args.cpu_budget:.0f}s reached ({spent:.0f}s)"
                break
            for kind in kinds:
                rows.append(attempt(task, kind, refs, scratch, not args.no_strict))
    finally:
        keep = os.path.join(out, "failed")
        for r in rows:
            if r.get("failed_log") and os.path.exists(r["failed_log"]):
                os.makedirs(keep, exist_ok=True)
                shutil.copy(r["failed_log"], keep)
        shutil.rmtree(scratch, ignore_errors=True)
    with open(os.path.join(out, "attempts.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    problems = list(ref_problems)
    if stopped:
        problems.append(stopped)
    problems += [
        f"{r['task']} {r['kind']}: {r['problems']}" for r in rows if not r["ok"]
    ]
    by_task: dict[str, set[str | None]] = {}
    for r in rows:
        by_task.setdefault(r["task"], set()).add(r["digest"])
    problems += [
        f"{t}: digests differ or missing"
        for t, d in by_task.items()
        if len(d) != 1 or None in d
    ]
    if len(rows) != 2 * len(tasks) * args.reps and not stopped:
        problems.append("incomplete matrix")
    summary = {
        "valid": not problems,
        "problems": problems,
        "environment": cp.environment(),
        "workload_sha256": cp.WORKLOAD_SHA256,
        "cpu_spent_s": sum(r["cpu_s"] for r in rows),
    }
    summary |= analyse([r for r in rows if r["ok"]])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(render(summary))
    return 0 if summary["valid"] else 1


def stats(values: list[float]) -> dict[str, float]:
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": cp.nearest_rank(values, 0.95),
        "min": min(values),
        "max": max(values),
        "iqr": q[2] - q[0],
    }


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traced = [r for r in rows if r["kind"] == "traced" and "breakdown" in r]
    plain = [r for r in rows if r["kind"] == "plain"]
    out: dict[str, Any] = {"attempts": {"traced": len(traced), "plain": len(plain)}}
    if not traced or not plain:
        return out
    out["overhead"] = {
        k: statistics.median(r[k] for r in traced)
        - statistics.median(r[k] for r in plain)
        for k in ("wall_s", "cpu_s")
    }
    out["plain_median"] = {
        k: statistics.median(r[k] for r in plain) for k in ("wall_s", "cpu_s")
    }
    out["traced_median"] = {
        k: statistics.median(r[k] for r in traced) for k in ("wall_s", "cpu_s")
    }
    out["inflated"] = out["overhead"]["wall_s"] > 0.1 * out["plain_median"]["wall_s"]

    def table(group: list[dict[str, Any]]) -> dict[str, Any]:
        crit = statistics.median(r["breakdown"]["critical_path_s"] for r in group)
        comps = {}
        for c in COMPONENTS:
            s = stats([r["breakdown"]["components"][c] for r in group])
            s["share"] = s["median"] / crit if crit else 0.0
            comps[c] = s
        return {"runs": len(group), "critical_path_median_s": crit, "components": comps}

    out["all"] = table(traced)
    loads = [r["cpu_pressure10"] for r in traced if r["cpu_pressure10"] is not None]
    if loads:
        cut = statistics.median(loads)
        low = [r for r in traced if (r["cpu_pressure10"] or 0) <= cut]
        high = [r for r in traced if (r["cpu_pressure10"] or 0) > cut]
        if low and high:
            out["low_load"], out["high_load"] = table(low), table(high)
            out["pressure_cut"] = cut
    out["per_call"] = {
        k: stats([r["breakdown"]["per_call"][k] for r in traced])
        for k in ("cgroup_probe", "sandbox_run", "cgroup_kill")
    }
    out["sandbox_calls_median"] = statistics.median(
        r["breakdown"]["sandbox_calls"] for r in traced
    )
    out["monitor"] = {
        "backend": traced[0]["breakdown"]["monitor"]["backend"],
        "samples_median": statistics.median(
            r["breakdown"]["monitor"]["samples"] for r in traced
        ),
        "sample_wall_s_median": statistics.median(
            r["breakdown"]["monitor"]["wall_s"] for r in traced
        ),
        "sample_cpu_s_median": statistics.median(
            r["breakdown"]["monitor"]["cpu_s"] for r in traced
        ),
    }
    procs: dict[str, list[float]] = {}
    for r in traced:
        for tag, v in r["breakdown"]["procs"].items():
            procs.setdefault(tag, []).append(v["life_s"])
    out["process_lifetimes_s"] = {t: stats(v) for t, v in sorted(procs.items())}
    out["by_load_ranking_agrees"] = ranking_agrees(out)
    return out


def ranking_agrees(doc: dict[str, Any]) -> bool | None:
    if "low_load" not in doc:
        return None

    def top(t: dict[str, Any]) -> list[str]:
        order = sorted(t["components"], key=lambda c: -t["components"][c]["median"])
        return order[:3]

    return top(doc["low_load"]) == top(doc["high_load"])


# ---------------------------------------------------------------------- micro


def _wait(cmd: list[str], env: dict[str, str] | None = None) -> tuple[float, float]:
    t0 = time.perf_counter()
    with open(os.devnull, "wb") as dn:
        p = subprocess.Popen(cmd, stdout=dn, stderr=dn, env=env, cwd=cp.ROOT)
        _, _, ru = os.wait4(p.pid, 0)
    return time.perf_counter() - t0, ru.ru_utime + ru.ru_stime


def micro_variants(
    work: str, script: str
) -> dict[str, Callable[[], tuple[float, float]]]:
    def probe() -> tuple[float, float]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PATH")
        }
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "-q",
            "-p",
            "MemoryMax=4G",
            "-p",
            "TasksMax=512",
            "true",
        ]
        return _wait(cmd, env)

    def bwrap_true() -> tuple[float, float]:
        return _wait(engine.bwrap_cmd(work, ["true"]))

    def scope_bwrap_true() -> tuple[float, float]:
        cmd, env = engine._build_sandbox_command(
            work, script, True, unit_name=f"bench-micro-{uuid.uuid4().hex[:10]}"
        )
        return _wait(cmd, env)

    def kill_absent() -> tuple[float, float]:
        t0, c0 = time.perf_counter(), time.process_time()
        engine._kill_cgroup_scope(f"bench-micro-{uuid.uuid4().hex[:10]}")
        return time.perf_counter() - t0, time.process_time() - c0

    def monitor_stop() -> tuple[float, float]:
        mon = telemetry.VramMonitor(interval_s=0.5)
        mon.start()
        time.sleep(0.05)  # inside the sampler's sleep, as in a real attempt
        t0 = time.perf_counter()
        mon.stop()
        return time.perf_counter() - t0, 0.0

    def sample() -> tuple[float, float]:
        t0, c0 = time.perf_counter(), time.process_time()
        telemetry.get_gpu_memory()
        return time.perf_counter() - t0, time.process_time() - c0

    return {
        "py_bare": lambda: _wait([sys.executable, "-I", "-c", "pass"]),
        "py_import_engine": lambda: _wait([sys.executable, "-c", "import bench.bench"]),
        "py_import_profile": lambda: _wait(
            [sys.executable, "-c", "import experiments.cpu_profile"]
        ),
        "probe": probe,
        "bwrap_true": bwrap_true,
        "scope_bwrap_true": scope_bwrap_true,
        "kill_absent": kill_absent,
        "monitor_sample": sample,
        "monitor_stop": monitor_stop,
    }


def importtime(reps: int = 5) -> list[dict[str, Any]]:
    """Cumulative import cost per module (median of `reps` -X importtime runs)."""
    seen: dict[str, list[float]] = {}
    for _ in range(reps):
        p = subprocess.run(
            [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                "import experiments.cpu_profile",
            ],
            cwd=cp.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        for line in p.stderr.splitlines():
            parts = line.split("|")
            if len(parts) == 3 and parts[1].strip().isdigit():
                seen.setdefault(parts[2].strip(), []).append(int(parts[1]) / 1e6)
    rows = [(statistics.median(v), m) for m, v in seen.items()]
    top = sorted(rows, reverse=True)[:12]
    return [{"module": m, "cumulative_s": c} for c, m in top]


def cmd_micro(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cpu-micro-", dir="/var/tmp")
    work = os.path.join(tmp, "w")
    os.makedirs(work)
    script = os.path.join(tmp, "run.sh")
    engine._write_runner_script(script, "true", True)
    variants = micro_variants(work, script)
    names = list(variants)
    samples: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    load = os.getloadavg()[0]
    try:
        for rep in range(args.reps):  # rotating order: no variant always runs first
            for n in names[rep % len(names) :] + names[: rep % len(names)]:
                samples[n].append(variants[n]())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    doc: dict[str, Any] = {"load1_at_start": load, "reps": args.reps, "variants": {}}
    for n, v in samples.items():
        doc["variants"][n] = {
            "wall_s": stats([w for w, _ in v]),
            "cpu_s": stats([c for _, c in v]),
        }
    doc["importtime"] = importtime()
    with open(os.path.join(out, "micro.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {n: round(d["wall_s"]["median"], 4) for n, d in doc["variants"].items()},
            indent=1,
        )
    )
    return 0


# ---------------------------------------------------------------------- report


def render(doc: dict[str, Any]) -> str:
    lines = [f"VALID: {doc['valid']}"] + [
        f"PROBLEM: {p}" for p in doc.get("problems", [])
    ]
    if "all" not in doc:
        return "\n".join(lines)
    lines.append(
        f"traced {doc['traced_median']['wall_s']:.3f}s vs plain "
        f"{doc['plain_median']['wall_s']:.3f}s wall (overhead "
        f"{doc['overhead']['wall_s']:+.3f}s, cpu {doc['overhead']['cpu_s']:+.3f}s)"
        + ("  INFLATED" if doc["inflated"] else "")
    )
    for label in ("all", "low_load", "high_load"):
        t = doc.get(label)
        if not t:
            continue
        lines.append(
            f"[{label}] {t['runs']} runs, critical path median "
            f"{t['critical_path_median_s']:.3f}s"
        )
        for c, s in sorted(t["components"].items(), key=lambda kv: -kv[1]["median"]):
            lines.append(
                f"  {c:14s} median {s['median']:.3f}  p95 {s['p95']:.3f}  "
                f"iqr {s['iqr']:.3f}  share {s['share']:.0%}"
            )
    lines.append(
        "ranking of top 3 agrees across load halves: "
        f"{doc.get('by_load_ranking_agrees')}"
    )
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    with open(os.path.join(args.dir, "summary.json"), encoding="utf-8") as f:
        doc = json.load(f)
    print(render(doc))
    return 0 if doc.get("valid") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", type=cp.count_arg, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--cpu-budget", type=cp.budget_arg, default=CPU_BUDGET_S)
    r.add_argument("--no-strict", action="store_true")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("micro")
    m.add_argument("--reps", type=cp.count_arg, default=30)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_micro)
    s = sub.add_parser("report")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
