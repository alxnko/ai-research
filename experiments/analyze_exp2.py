#!/usr/bin/env python3
"""EXP-2 analysis: paired Qwen verification-policy study.

Reads runs/exp2/{control,verification}/seed{42,43,44}/<task>/results.json and
prints the registered outcome and secondary metrics. Read-only.

Usage: uv run python experiments/analyze_exp2.py [--json]
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "exp2")
ARMS = ("control", "verification")
SEEDS = (42, 43, 44)
TASKS = (
    "backend-auth-token",
    "backend-idempotency",
    "debugging-off-by-one",
    "debugging-money-round",
    "refactor-parser",
    "refactor-validators",
)


def wilson(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load() -> dict[tuple[str, int, str], dict]:
    records = {}
    for path in glob.glob(os.path.join(RUNS, "*", "seed*", "*", "results.json")):
        arm = path.split(os.sep)[-4]
        seed = int(path.split(os.sep)[-3].removeprefix("seed"))
        task = path.split(os.sep)[-2]
        if arm in ARMS and task in TASKS:
            with open(path) as f:
                rows = json.load(f)
            records[(arm, seed, task)] = rows[0] if rows else {}
    return records


def summarize(records, arm: str) -> dict:
    rows = [
        records[(arm, s, t)] for s in SEEDS for t in TASKS if (arm, s, t) in records
    ]
    ok = [r for r in rows if r.get("run_successful")]
    wall = sum(r.get("total_wall_s") or 0.0 for r in rows)
    evals = sum(r.get("tokens") or 0 for r in rows if not r.get("tokens_incomplete"))
    return {
        "arm": arm,
        "attempts": len(rows),
        "successes": len(ok),
        "success_rate": len(ok) / len(rows) if rows else None,
        "wilson95": wilson(len(ok), len(rows)),
        "total_wall_s": round(wall, 1),
        "verified_tasks_per_hour": round(3600 * len(ok) / wall, 2) if wall else None,
        "median_wall_s": round(
            statistics.median([r.get("total_wall_s") or 0.0 for r in rows]), 1
        )
        if rows
        else None,
        "p95_wall_s": round(
            sorted(r.get("total_wall_s") or 0.0 for r in rows)[
                max(0, math.ceil(0.95 * len(rows)) - 1)
            ],
            1,
        )
        if rows
        else None,
        "tokens_evaluated": evals,
        "tokens_incomplete_runs": sum(1 for r in rows if r.get("tokens_incomplete")),
        "inference_errors": sum(1 for r in rows if r.get("inference_error")),
        "timeouts": sum(1 for r in rows if r.get("timed_out")),
        "invalid_actions": sum(r.get("invalid_actions") or 0 for r in rows),
        "peak_vram_mib": max((r.get("peak_vram_mib") or 0 for r in rows), default=0),
    }


def main() -> int:
    records = load()
    missing = [
        (a, s, t)
        for a in ARMS
        for s in SEEDS
        for t in TASKS
        if (a, s, t) not in records
    ]
    stats = {arm: summarize(records, arm) for arm in ARMS}

    if "--json" in sys.argv:
        json.dump({"arms": stats, "missing": missing}, sys.stdout, indent=1)
        print()
        return 0

    print(f"attempts found: {len(records)}/36")
    if missing:
        print(f"missing: {len(missing)}")
        for a, s, t in missing[:6]:
            print(f"  {a} seed{s} {t}")

    for arm in ARMS:
        d = stats[arm]
        lo, hi = d["wilson95"]
        print(
            f"\n{arm:13s} {d['successes']}/{d['attempts']} "
            f"({(d['success_rate'] or 0) * 100:.1f}%, 95% CI "
            f"{lo * 100:.0f}-{hi * 100:.0f}%)  "
            f"{d['verified_tasks_per_hour']} tasks/h over {d['total_wall_s']}s"
        )
        print(
            f"  median {d['median_wall_s']}s  p95 {d['p95_wall_s']}s  "
            f"tokens {d['tokens_evaluated']} (incomplete runs "
            f"{d['tokens_incomplete_runs']})"
        )
        print(
            f"  inference errors {d['inference_errors']}  timeouts "
            f"{d['timeouts']}  invalid actions {d['invalid_actions']}  "
            f"peak VRAM {d['peak_vram_mib']} MiB"
        )

    print("\ntask x arm successes (of 3):")
    for task in TASKS:
        row = []
        for arm in ARMS:
            wins = sum(
                1
                for s in SEEDS
                if records.get((arm, s, task), {}).get("run_successful")
            )
            row.append(f"{wins}/3")
        print(f"  {task:24s} control {row[0]}  treatment {row[1]}")

    paired = [
        k
        for k in records
        if k[0] == "control" and ("verification", k[1], k[2]) in records
    ]

    def ok(arm, key):
        return bool(records.get((arm, key[1], key[2]), {}).get("run_successful"))

    t_wins = sum(1 for k in paired if ok("verification", k) and not ok("control", k))
    c_wins = sum(1 for k in paired if ok("control", k) and not ok("verification", k))
    print(f"\npaired discordance: treatment-only {t_wins}, control-only {c_wins}")

    c, t = stats["control"], stats["verification"]
    if c["verified_tasks_per_hour"] and t["verified_tasks_per_hour"]:
        ratio = t["verified_tasks_per_hour"] / c["verified_tasks_per_hour"]
        winner = (
            ratio >= 1.25
            and t["successes"] >= c["successes"]
            and not (t["inference_errors"] > c["inference_errors"])
            and not (t["timeouts"] > c["timeouts"])
        )
        print(f"throughput ratio treatment/control: {ratio:.2f}x")
        print(f"registered decision rule met: {winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
