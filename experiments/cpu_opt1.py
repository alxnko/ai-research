"""CPU-OPT-1 runner: control vs the registered one-line treatment, interleaved.

Protocol: `CPU-OPT-1_PROTOCOL.md`. Both arms are git worktrees of one commit and
run through the same driver (`experiments.cpu_profile one`, measured from outside
with `wait4`); the treatment differs by `cpu_opt1_treatment.patch` and nothing
else. No GPU work, no model. Calibration tasks are a dry run: no decision is made.
The final set needs `--registered-final` and is used once.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from experiments import cpu_profile as cp
from experiments import monitor_equiv
from experiments.cpu_breakdown import pressure

PATCH = os.path.join(cp.ROOT, "experiments", "cpu_opt1_treatment.patch")
TELEMETRY = "bench/telemetry.py"
PINS = {  # relative path -> sha256, identical in both arms
    "bench/bench.py": (
        "6e7ffb18189df5abdbf09bf115bb2702b55b2d1076a5117b89fc5a5ccf0d3b5a"
    ),
    "bench/classifier.py": (
        "a8c2b9b6a672166c02bfd66ba6fc89f8f59d52c8f590afb7fad918194913158c"
    ),
}
CONTROL_TELEMETRY = "b570ab8fe11a8843c17788ce95362f9bcc9cd9b3e7cdedc7a53b75f10b01dc00"
TREATMENT_TELEMETRY = "0b8b001b5874c15008c4d0b8be6e689e731079fed06863ddaee789bb159cae1f"
CHANGED = (
    "-            time.sleep(self.interval_s)",
    "+            self.stop_event.wait(self.interval_s)",
)

SEED = 20261118
RESAMPLES = 10_000
MIN_IMPROVEMENT_S = 0.20
MIN_LOWER_BOUND_S = 0.10
MAX_CPU_REGRESSION_S = 0.02
CPU_BUDGET_S = 600.0
REPS = 10
MIN_HALF = 20  # pairs required in each load group for the halves to be evaluable
ARMS = ("control", "treatment")
RESERVATION_DIR = os.path.join(cp.ROOT, "runs", "cpu-opt1")
RESERVED = "final-run.json"  # exclusive: created once, by the first final run
RECOVERY = "final-recovery.json"  # exclusive: the single infrastructure recovery


# ------------------------------------------------------------------------ arms


def git(root: str, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=True
    )
    return done.stdout


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def arm_problems(
    control: str,
    treatment: str,
    pins: dict[str, str] | None = None,
    telemetry: tuple[str, str] = (CONTROL_TELEMETRY, TREATMENT_TELEMETRY),
) -> list[str]:
    """Nothing may differ between the arms except the registered one-line change."""
    pins = PINS if pins is None else pins
    problems = []
    for rel, want in pins.items():
        for label, root in (("control", control), ("treatment", treatment)):
            if sha256_file(os.path.join(root, rel)) != want:
                problems.append(f"{label} {rel} is not the pinned file")
    for label, root, want in (
        ("control", control, telemetry[0]),
        ("treatment", treatment, telemetry[1]),
    ):
        if sha256_file(os.path.join(root, TELEMETRY)) != want:
            problems.append(f"{label} {TELEMETRY} is not the pinned file")
    if git(control, "status", "--porcelain").strip():
        problems.append("control worktree is not identical to its commit")
    if git(control, "rev-parse", "HEAD") != git(treatment, "rev-parse", "HEAD"):
        problems.append("the arms are not worktrees of one commit")
    names = git(treatment, "diff", "--name-only").split()
    status = git(treatment, "status", "--porcelain").splitlines()
    if names != [TELEMETRY] or len(status) != 1:
        problems.append(f"treatment differs from control in more than {TELEMETRY}")
    numstat = git(treatment, "diff", "--numstat").split()
    if numstat[:3] != ["1", "1", TELEMETRY]:
        problems.append(f"treatment diff is not 1 insertion and 1 deletion: {numstat}")
    body = [
        ln
        for ln in git(treatment, "diff", "-U0").splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    if tuple(body) != CHANGED:
        problems.append(f"treatment diff is not the registered change: {body}")
    return problems


def make_arms(base: str, *, allow_dirty: bool = False) -> tuple[str, str]:
    """Two worktrees of HEAD under `base`; the treatment gets the patch."""
    if not allow_dirty and cp.environment()["tree_dirty"]:
        raise RuntimeError("working tree is not clean: commit before running")
    control, treatment = (os.path.join(base, n) for n in ("control", "treatment"))
    for root in (control, treatment):
        git(cp.ROOT, "worktree", "add", "--detach", root, "HEAD")
    git(treatment, "apply", PATCH)
    return control, treatment


def drop_arms(*roots: str) -> None:
    for root in roots:
        subprocess.run(
            ["git", "-C", cp.ROOT, "worktree", "remove", "--force", root],
            capture_output=True,
        )
    subprocess.run(["git", "-C", cp.ROOT, "worktree", "prune"], capture_output=True)


# -------------------------------------------------------------------- attempts


@dataclasses.dataclass(frozen=True)
class Context:
    refs: dict[str, Any]
    scratch: str
    strict: bool


def attempt(arm: str, root: str, task: str, ctx: Context) -> dict[str, Any]:
    refs, scratch, strict = ctx.refs, ctx.scratch, ctx.strict
    name = f"{task}-{arm}-{uuid.uuid4().hex[:6]}"
    run_dir, log = os.path.join(scratch, name), os.path.join(scratch, f"{name}.log")
    cmd = [sys.executable, "-m", "experiments.cpu_profile", "one"]
    cmd += ["--task", task, "--run-dir", run_dir]
    if not strict:
        cmd.append("--no-strict-isolation")
    outer, exit_code, timed_out = cp.measure_command(
        cmd, log, cp.CHILD_TIMEOUT_S, cwd=root
    )
    row = cp.read_row(run_dir)
    evidence = cp.disk_evidence(run_dir, task)
    problems = cp.integrity_problems(row, evidence, refs.get(task))
    if exit_code != 0 or timed_out:
        problems.append(f"child exit {exit_code}, timed_out={timed_out}")
    samples = (row or {}).get("monitor_sample_count")
    if not isinstance(samples, int) or samples < 1:
        problems.append("the GPU monitor did not sample: the change is not exercised")
    return {
        "task": task,
        "arm": arm,
        **outer,
        "cpu_s": outer["user_s"] + outer["sys_s"],
        "digest": cp.canonical_digest({**(row or {}), **evidence}),
        "samples": samples,
        "ok": not problems,
        "problems": problems,
        "failed_log": log if problems else None,
    }


def pair_order(rep: int, index: int) -> list[str]:
    return (
        ["control", "treatment"] if (rep + index) % 2 == 0 else ["treatment", "control"]
    )


def matrix_problems(
    rows: list[dict[str, Any]], tasks: list[str], reps: int
) -> list[str]:
    """Every registered (task, repetition, arm) cell exactly once, nothing else."""
    expected = collections.Counter(
        (t, r, a) for t in tasks for r in range(reps) for a in ARMS
    )
    actual = collections.Counter((r["task"], r["rep"], r["arm"]) for r in rows)
    problems = []
    if missing := sorted(expected - actual):
        problems.append(f"missing cells: {missing}")
    if dup := sorted(k for k, n in actual.items() if k in expected and n > 1):
        problems.append(f"duplicate cells: {dup}")
    if extra := sorted(k for k in actual if k not in expected):
        problems.append(f"unexpected cells: {extra}")
    return problems


def validity_problems(
    rows: list[dict[str, Any]], tasks: list[str], reps: int, budget_s: float
) -> list[str]:
    """Correctness gate. An invalid run gets no effect statistics at all."""
    problems = [
        f"{r['task']} {r['arm']} rep {r['rep']}: {r['problems']}"
        for r in rows
        if not r["ok"]
    ]
    problems += matrix_problems(rows, tasks, reps)
    digests: dict[str, set[str | None]] = {}
    for r in rows:
        digests.setdefault(r["task"], set()).add(r["digest"])
    problems += [
        f"{t}: digests differ between runs/arms or are missing"
        for t, d in digests.items()
        if len(d) != 1 or None in d
    ]
    spent = sum(r["cpu_s"] for r in rows)
    if spent > budget_s:
        problems.append(f"cpu budget {budget_s:.0f}s overshot ({spent:.0f}s)")
    by_key: dict[tuple[str, int], dict[str, int | None]] = {}
    for r in rows:
        by_key.setdefault((r["task"], r["rep"]), {})[r["arm"]] = r["samples"]
    gaps = [
        c - t
        for p in by_key.values()
        if isinstance((c := p.get("control")), int)
        and isinstance((t := p.get("treatment")), int)
    ]
    if gaps and (statistics.median(gaps) != 0 or max(abs(g) for g in gaps) > 1):
        problems.append(f"monitor sample counts differ between arms: {sorted(gaps)}")
    return problems


# -------------------------------------------------------------------- analysis


def pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    got: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        got.setdefault((r["task"], r["rep"]), {})[r["arm"]] = r
    out = []
    for (task, rep), p in sorted(got.items()):
        c, t = p["control"], p["treatment"]
        out.append(
            {
                "task": task,
                "rep": rep,
                "d": c["wall_s"] - t["wall_s"],
                "cpu": t["cpu_s"] - c["cpu_s"],
                "pressure": p["control"].get("pressure"),
            }
        )
    return out


def stratified_interval(
    by_task: dict[str, list[float]], seed: int = SEED, n: int = RESAMPLES
) -> tuple[float, float]:
    """95% percentile interval for the median of all pairs: in each resample the
    pairs of every task are drawn with replacement within that task."""
    rng = random.Random(seed)
    meds = sorted(
        statistics.median(
            x for ds in by_task.values() for x in rng.choices(ds, k=len(ds))
        )
        for _ in range(n)
    )
    return meds[int(0.025 * n)], meds[math.ceil(0.975 * n) - 1]


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ps = pairs(rows)
    by_task: dict[str, list[float]] = {}
    for p in ps:
        by_task.setdefault(p["task"], []).append(p["d"])
    lower, upper = stratified_interval(by_task)
    out: dict[str, Any] = {
        "pairs": len(ps),
        "median_d_s": statistics.median(p["d"] for p in ps),
        "interval_95_s": [lower, upper],
        "p95_d_s": cp.nearest_rank([p["d"] for p in ps], 0.95),
        "per_task_median_d_s": {
            t: statistics.median(v) for t, v in sorted(by_task.items())
        },
        "median_cpu_regression_s": statistics.median(p["cpu"] for p in ps),
        "control_median_wall_s": statistics.median(
            r["wall_s"] for r in rows if r["arm"] == "control"
        ),
        "treatment_median_wall_s": statistics.median(
            r["wall_s"] for r in rows if r["arm"] == "treatment"
        ),
    }
    # Registered tie policy: cut = median pressure; low is strictly below it, high
    # strictly above, pairs at the cut belong to neither. If either group has fewer
    # than MIN_HALF pairs (for example an idle host reading 0.00 throughout) the
    # load groups are not evaluable and no high-load group is manufactured.
    halves = None
    if all(p["pressure"] is not None for p in ps):
        cut = statistics.median(p["pressure"] for p in ps)
        low = [p["d"] for p in ps if p["pressure"] < cut]
        high = [p["d"] for p in ps if p["pressure"] > cut]
        if len(low) >= MIN_HALF and len(high) >= MIN_HALF:
            halves = {
                "cut": cut,
                "low_pairs": len(low),
                "high_pairs": len(high),
                "low": statistics.median(low),
                "high": statistics.median(high),
            }
    out["load_halves"] = halves  # None: not evaluable
    return out


def decide(stats: dict[str, Any]) -> str:
    """The registered decision rule, applied once to a valid, complete run."""
    halves = stats["load_halves"]
    if halves is None or (halves["low"] > 0) != (halves["high"] > 0):
        return "INCONCLUSIVE DUE TO LOAD"
    met = (
        stats["median_d_s"] >= MIN_IMPROVEMENT_S
        and stats["interval_95_s"][0] > MIN_LOWER_BOUND_S
        and stats["median_cpu_regression_s"] <= MAX_CPU_REGRESSION_S
    )
    return "ADOPT AS v0.6 CANDIDATE" if met else "NEGATIVE: minimum improvement not met"


# ------------------------------------------------------------------------- run


class Recorder:
    """Appends each attempt durably as it completes and keeps a failed attempt's
    log at once, so a crash later cannot lose completed evidence."""

    def __init__(self, out: str):
        self.rows: list[dict[str, Any]] = []
        self.out = out
        self.path = os.path.join(out, "attempts.jsonl")

    def __call__(self, row: dict[str, Any]) -> None:
        log = row.get("failed_log")
        if log and os.path.exists(log):
            os.makedirs(os.path.join(self.out, "failed"), exist_ok=True)
            shutil.copy(log, os.path.join(self.out, "failed"))
            row = {**row, "failed_log": os.path.join("failed", os.path.basename(log))}
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def refusal(args: argparse.Namespace, out: str) -> str | None:
    """Reasons to refuse before touching anything."""
    if any(
        os.path.exists(os.path.join(out, n)) for n in ("summary.json", "attempts.jsonl")
    ):
        return f"{out} already holds records; choose a new --out"
    if args.set != "final":
        return "--recover-of applies to the final run only" if args.recover_of else None
    for bad, why in (
        (not args.registered_final, "the final set needs --registered-final"),
        (args.reps != REPS, f"the registered final run uses exactly {REPS} reps"),
        (args.no_strict, "the registered final run uses strict isolation"),
        (
            args.cpu_budget != CPU_BUDGET_S,
            f"the registered final run uses exactly {CPU_BUDGET_S:.0f} CPU-seconds",
        ),
    ):
        if bad:
            return why
    return None


def read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError, ValueError:
        return None


def claim(name: str, record: dict[str, Any]) -> bool:
    """Create a reservation file exclusively; False when it already exists."""
    os.makedirs(RESERVATION_DIR, exist_ok=True)
    try:
        fd = os.open(
            os.path.join(RESERVATION_DIR, name),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    return True


def recovery_problem(original: str) -> str | None:
    """A run may be recovered only if it never produced a result: it crashed or
    aborted from a cause outside both arms. A completed run, a gate failure, a
    digest or hash mismatch, or an effect statistic is never recoverable."""
    first = read_json(os.path.join(RESERVATION_DIR, RESERVED))
    if first is None or first.get("out") != os.path.abspath(original):
        return "--recover-of must name the run that holds the final reservation"
    summary = read_json(os.path.join(original, "summary.json"))
    if summary is None:
        if os.path.exists(os.path.join(original, "started.json")):
            return None  # killed before any summary: a crash
        return "the original run left no started record"
    if "stats" in summary or summary.get("valid"):
        return "the original run produced a result; it is not recoverable"
    if "aborted" not in summary:
        return "the original run failed a gate or check; that is a result, not infra"
    return None


def reserve_final(out: str, recover_of: str | None) -> str | None:
    """Persistent, exclusive claim on the one registered final run."""
    record = {"out": out, "recover_of": recover_of, "time": time.time()}
    if recover_of is None:
        if claim(RESERVED, record):
            return None
        first = read_json(os.path.join(RESERVATION_DIR, RESERVED)) or {}
        return f"the registered final run was already started: {first.get('out')}"
    if problem := recovery_problem(recover_of):
        return problem
    if not claim(RECOVERY, record):
        return "the single infrastructure recovery was already used"
    return None


def run_matrix(
    args: argparse.Namespace, out: str, tasks: list[str], base: str, rec: Recorder
) -> list[str]:
    """Preflight, reserve (final), measure. Preflight failures do not consume the
    final reservation: nothing has been measured yet."""
    arms = make_arms(os.path.join(base, "arms"))
    problems = arm_problems(*arms) + monitor_equiv.equivalence_problems(
        os.path.join(arms[0], TELEMETRY), os.path.join(arms[1], TELEMETRY)
    )
    refs, ref_problems = cp.build_references(tasks)
    if problems or ref_problems:
        return problems + ref_problems
    if args.set == "final" and (err := reserve_final(out, args.recover_of)):
        return [err]
    with open(os.path.join(out, "started.json"), "w", encoding="utf-8") as f:
        json.dump({"environment": cp.environment(), "set": args.set}, f)
    scratch = os.path.join(base, "scratch")
    os.makedirs(scratch)
    ctx = Context(refs, scratch, not args.no_strict)
    stopped = measure(arms, tasks, ctx, args, rec)
    return [stopped] if stopped else []


def cmd_run(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.out)
    if error := refusal(args, out):
        print(error, file=sys.stderr)
        return 2
    try:
        tasks = cp.workload()[args.set]
    except ValueError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    base = tempfile.mkdtemp(prefix="cpu-opt1-", dir="/var/tmp")
    rec = Recorder(out)
    aborted = None
    try:
        problems = run_matrix(args, out, tasks, base, rec)
    except Exception as exc:  # a runner fault, not a result: recorded, run invalid
        aborted = f"{type(exc).__name__}: {exc}"
        problems = [f"aborted: {aborted}"]
    finally:
        drop_arms(*(os.path.join(base, "arms", n) for n in ARMS))
        shutil.rmtree(base, ignore_errors=True)
    problems += validity_problems(rec.rows, tasks, args.reps, args.cpu_budget)
    final = args.set == "final"
    summary: dict[str, Any] = {
        "valid": not problems,
        "problems": problems,
        "set": args.set,
        "dry_run": not final,
        "recovery_of": args.recover_of,
        "environment": cp.environment(),
        "workload_sha256": cp.WORKLOAD_SHA256,
        "cpu_spent_s": sum(r["cpu_s"] for r in rec.rows),
    }
    if aborted:
        summary["aborted"] = aborted
    if not problems:  # no effect statistic is ever computed on an invalid run
        summary["stats"] = analyse(rec.rows)
        summary["decision"] = (
            decide(summary["stats"]) if final else "DRY RUN: not part of the decision"
        )
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "environment"}, indent=2)
    )
    return 0 if summary["valid"] else 1


def measure(arms, tasks, ctx, args, record) -> str | None:
    """Run the pairs, recording each attempt as it completes. Returns why it
    stopped early, or None when the whole matrix ran."""
    roots = dict(zip(ARMS, arms, strict=True))
    for rep in range(args.reps):
        for i, task in enumerate(tasks):
            spent = sum(r["cpu_s"] for r in record.rows)
            worst = max((r["cpu_s"] for r in record.rows), default=0.0)
            if spent + 2 * worst >= args.cpu_budget:  # admission control only
                return f"cpu budget {args.cpu_budget:.0f}s: stopped admitting pairs"
            host = pressure()  # one reading per pair, shared by both arms
            for arm in pair_order(rep, i):
                row = attempt(arm, roots[arm], task, ctx)
                record({**row, "rep": rep, "pressure": host})
                if not row["ok"]:  # registered stop condition: no further attempts
                    return f"stopped after a failed attempt: {task} {arm} rep {rep}"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=["calibration", "final"], default="calibration")
    r.add_argument("--reps", type=cp.count_arg, default=REPS)
    r.add_argument("--out", required=True)
    r.add_argument("--cpu-budget", type=cp.budget_arg, default=CPU_BUDGET_S)
    r.add_argument("--registered-final", action="store_true")
    r.add_argument("--no-strict", action="store_true")
    r.add_argument("--recover-of", help="the one infrastructure recovery (final)")
    r.set_defaults(fn=cmd_run)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
