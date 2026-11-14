#!/usr/bin/env python3
"""Manifest-driven experiment controller.

One manifest names the experiment: benchmark version and suite, tasks, seeds,
arms (with prompt hashes), model identity and run settings. The controller
derives the full attempt matrix from it, validates every recorded attempt
against it (provenance, not just "a results.json exists"), runs missing
attempts through the existing `bench/bench.py` engine without touching it, and
analyses the result with explicit handling of zero-success arms.

    uv run python -m experiments.controller check   MANIFEST
    uv run python -m experiments.controller status  MANIFEST [--runs DIR]
    uv run python -m experiments.controller analyze MANIFEST [--runs DIR] [--json]
    uv run python -m experiments.controller plan    MANIFEST [--runs DIR]
    uv run python -m experiments.controller run     MANIFEST

Everything except `run` is offline: no Ollama, GPU or network.
`analyze` enforces what the frozen EXP-2 analyzer only reports: it exits
non-zero while any attempt is missing, invalid or unexpected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

from bench import bench as engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = os.path.join(ROOT, "bench", "suites")
REQUIRED_ROW = ("task", "passed", "run_successful", "total_wall_s", "tokens")
QUARANTINE = (".partial", ".aborted-", ".invalid-")

# key -> expected type (unknown keys are rejected, so typos fail loudly)
_TOP: dict[str, type] = {
    "schema": int,
    "experiment": str,
    "frozen": bool,
    "benchmark": dict,
    "tasks": list,
    "seeds": list,
    "arms": dict,
    "model": dict,
    "settings": dict,
    "runs_dir": str,
    "provenance_file": str,
    "aborted": list,
    "analysis": dict,
}
_BENCHMARK: dict[str, type] = {"tag": str, "commit": str, "suite": str}
_MODEL: dict[str, type] = {
    "tag": str,
    "manifest_digest": str,
    "config_digest": str,
    "weights_digest": str,
}
_SETTINGS: dict[str, type] = {
    "actuator": str,
    "ctx": int,
    "ngpu": int,
    "steps": int,
    "max_wall": int,
    "strict_isolation": bool,
    "sampling": dict,
}
_SAMPLING: dict[str, type] = {
    "profile": str,
    "version": str,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "repeat_penalty": float,
}


def _typed(obj: Any, spec: dict[str, type], where: str, errors: list[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{where}: must be an object")
        return
    for key, kind in spec.items():
        value = obj.get(key)
        if key not in obj:
            errors.append(f"{where}.{key}: missing")
        elif not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            errors.append(f"{where}.{key}: must be {kind.__name__}")
    for key in obj:
        if key not in spec:
            errors.append(f"{where}.{key}: unknown key")


def validate_manifest(m: Any) -> list[str]:
    """Schema problems of a manifest (empty list: valid)."""
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["manifest must be an object"]
    required = set(_TOP) - {"aborted", "frozen", "provenance_file"}
    for key in sorted(required - set(m)):
        errors.append(f"{key}: missing")
    for key in m:
        if key not in _TOP:
            errors.append(f"{key}: unknown key")
        elif not isinstance(m[key], _TOP[key]) or (
            _TOP[key] is int and isinstance(m[key], bool)
        ):
            errors.append(f"{key}: must be {_TOP[key].__name__}")
    if errors:
        return errors
    if m["schema"] != 1:
        errors.append("schema: only 1 is supported")
    _typed(m["benchmark"], _BENCHMARK, "benchmark", errors)
    _typed(m["model"], _MODEL, "model", errors)
    _typed(m["settings"], _SETTINGS, "settings", errors)
    _typed(m["settings"].get("sampling"), _SAMPLING, "settings.sampling", errors)
    for field in ("tasks", "seeds"):
        items = m[field]
        if not items or len(set(items)) != len(items):
            errors.append(f"{field}: must be non-empty and unique")
    if not all(isinstance(t, str) for t in m["tasks"]):
        errors.append("tasks: must be strings")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in m["seeds"]):
        errors.append("seeds: must be integers")
    if not m["arms"]:
        errors.append("arms: must not be empty")
    for name, arm in m["arms"].items():
        digest = arm.get("prompt_sha256") if isinstance(arm, dict) else None
        if not (isinstance(digest, str) and len(digest) == 64):
            errors.append(f"arms.{name}.prompt_sha256: must be a sha256 hex digest")
        extra = set(arm) - {"prompt_sha256", "launch"} if isinstance(arm, dict) else ()
        if extra:
            errors.append(f"arms.{name}: unknown keys {sorted(extra)}")
    analysis = m["analysis"]
    if set(analysis) != {"control", "treatment", "min_ratio"}:
        errors.append("analysis: needs exactly control, treatment, min_ratio")
    elif not {analysis["control"], analysis["treatment"]} <= set(m["arms"]):
        errors.append("analysis: control/treatment must name arms")
    return errors


def load_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(f"invalid manifest {path}: " + "; ".join(problems))
    return manifest


def manifest_sha256(manifest: dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def attempts(manifest: dict[str, Any]) -> Iterator[tuple[str, int, str]]:
    """The full matrix in execution order: arm, then seed, then task."""
    for arm in manifest["arms"]:
        for seed in manifest["seeds"]:
            for task in manifest["tasks"]:
                yield arm, seed, task


def attempt_id(manifest: dict[str, Any], arm: str, seed: int, task: str) -> str:
    """Stable identity: a different manifest gives different ids."""
    text = f"{manifest_sha256(manifest)}:{arm}:{seed}:{task}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cell_dir(runs: str, arm: str, seed: int, task: str) -> str:
    return os.path.join(runs, arm, f"seed{seed}", task)


def bench_argv(manifest: dict[str, Any], seed: int, task: str) -> list[str]:
    """The exact bench.py arguments of one attempt (same order EXP-2 recorded)."""
    s, sampling = manifest["settings"], manifest["settings"]["sampling"]
    argv = [
        "--model",
        manifest["model"]["tag"],
        "--actuator",
        s["actuator"],
        "--task",
        task,
        "--ctx",
        str(s["ctx"]),
        "--ngpu",
        str(s["ngpu"]),
        "--steps",
        str(s["steps"]),
        "--max-wall",
        str(s["max_wall"]),
    ]
    if s["strict_isolation"]:
        argv.append("--strict-isolation")
    argv += [
        "--sampling-profile",
        sampling["profile"],
        "--temperature",
        str(sampling["temperature"]),
        "--top-p",
        str(sampling["top_p"]),
        "--top-k",
        str(sampling["top_k"]),
        "--repeat-penalty",
        str(sampling["repeat_penalty"]),
        "--seed",
        str(seed),
    ]
    return argv


def _suite_pins(manifest: dict[str, Any]) -> dict[str, str]:
    name = manifest["benchmark"]["suite"]
    with open(os.path.join(SUITES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)["tasks"]


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_attempt(
    manifest: dict[str, Any], arm: str, seed: int, task: str, cell: str
) -> list[str]:
    """Why a recorded attempt is not trustworthy (empty list: valid).

    Checks the record against the manifest: task fixture digest, model
    identity, prompt hash, settings, sampling seed and the exact bench
    invocation, not merely that a results.json exists.
    """
    problems: list[str] = []
    provenance_name = manifest.get("provenance_file", "provenance.json")
    docs: dict[str, Any] = {}
    for name in ("config.json", "results.json", provenance_name):
        try:
            docs[name] = _read_json(os.path.join(cell, name))
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: unreadable ({type(exc).__name__})")
    if problems:
        return problems

    def want(ok: bool, text: str) -> None:
        if not ok:
            problems.append(text)

    rows = docs["results.json"]
    row = rows[0] if isinstance(rows, list) and len(rows) == 1 else None
    want(row is not None, "results.json must hold exactly one record")
    settings, sampling = manifest["settings"], dict(manifest["settings"]["sampling"])
    sampling_seed = {**sampling, "seed": seed}
    if isinstance(row, dict):
        missing = [k for k in REQUIRED_ROW if k not in row]
        want(not missing, f"results.json missing keys {missing}")
        want(row.get("task") == task, f"results task {row.get('task')!r} != {task!r}")
        want(
            row.get("task_digest") == _suite_pins(manifest).get(task),
            "task fixture digest differs from the suite pin",
        )
        want(
            isinstance(row.get("run_successful"), bool),
            "run_successful must be a boolean",
        )
        want(
            row.get("strict_isolation") == settings["strict_isolation"],
            "strict_isolation differs from the manifest",
        )
        want(row.get("actuator") == settings["actuator"], "actuator differs")
        want(row.get("sampling") == sampling_seed, "sampling differs from manifest")
        want(
            isinstance(row.get("total_wall_s"), int | float)
            and not isinstance(row.get("total_wall_s"), bool)
            and row["total_wall_s"] >= 0,
            "total_wall_s must be a non-negative number",
        )
    config = docs["config.json"]
    for key in ("ctx", "ngpu", "steps", "max_wall", "actuator", "strict_isolation"):
        want(config.get(key) == settings[key], f"config.{key} differs from manifest")
    want(config.get("model") == manifest["model"]["tag"], "config.model differs")
    want(config.get("task") == task, "config.task differs")
    want(config.get("sampling") == sampling_seed, "config.sampling differs")

    prov = docs[provenance_name]
    arm_spec = manifest["arms"][arm]
    bench = manifest["benchmark"]
    want(prov.get("policy") == arm, f"policy {prov.get('policy')!r} != arm {arm!r}")
    want(
        prov.get("prompt_sha256") == arm_spec["prompt_sha256"],
        "prompt hash differs from the arm's registered prompt",
    )
    want(prov.get("benchmark_tag") == bench["tag"], "benchmark tag differs")
    want(prov.get("benchmark_core_commit") == bench["commit"], "core commit differs")
    fingerprint = prov.get("model_fingerprint") or {}
    for key in ("manifest_digest", "config_digest", "weights_digest"):
        want(fingerprint.get(key) == manifest["model"][key], f"model {key} differs")
    want(
        prov.get("benchmark_argv") == bench_argv(manifest, seed, task),
        "recorded bench invocation differs from the manifest's",
    )
    return problems


def _is_quarantine(name: str) -> bool:
    return any(mark in name for mark in QUARANTINE)


def scan(manifest: dict[str, Any], runs: str) -> dict[str, Any]:
    """State of every cell of the matrix, plus anything outside it."""
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for arm, seed, task in attempts(manifest):
        cell = cell_dir(runs, arm, seed, task)
        if not os.path.exists(os.path.join(cell, "results.json")):
            state: dict[str, Any] = {"state": "missing", "problems": []}
        else:
            problems = validate_attempt(manifest, arm, seed, task, cell)
            state = {"state": "invalid" if problems else "valid", "problems": problems}
        cells[(arm, seed, task)] = state

    expected = {cell_dir(runs, *key) for key in cells}
    unexpected: list[str] = []
    quarantined: list[str] = []
    for arm in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        arm_dir = os.path.join(runs, arm)
        if not os.path.isdir(arm_dir):
            continue
        for seed_name in sorted(os.listdir(arm_dir)):
            seed_dir = os.path.join(arm_dir, seed_name)
            if not (seed_name.startswith("seed") and os.path.isdir(seed_dir)):
                continue
            for name in sorted(os.listdir(seed_dir)):
                path = os.path.join(seed_dir, name)
                if path in expected:
                    continue
                (quarantined if _is_quarantine(name) else unexpected).append(path)
    return {"cells": cells, "unexpected": unexpected, "quarantined": quarantined}


def wilson(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson interval; same definition as the frozen EXP-2 analyzer."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("run_successful")]
    wall = sum(r.get("total_wall_s") or 0.0 for r in rows)
    walls = sorted(r.get("total_wall_s") or 0.0 for r in rows)
    return {
        "attempts": len(rows),
        "successes": len(ok),
        "outcome": "no_success" if rows and not ok else "has_success",
        "success_rate": len(ok) / len(rows) if rows else None,
        "wilson95": wilson(len(ok), len(rows)),
        "total_wall_s": round(wall, 1),
        # zero successes is a measured 0.0 tasks/h, never a missing value
        "verified_tasks_per_hour": round(3600 * len(ok) / wall, 2) if wall else None,
        "median_wall_s": round(statistics.median(walls), 1) if walls else None,
        "p95_wall_s": (
            round(walls[max(0, math.ceil(0.95 * len(walls)) - 1)], 1) if walls else None
        ),
        "tokens_evaluated": sum(
            r.get("tokens") or 0 for r in rows if not r.get("tokens_incomplete")
        ),
        "tokens_incomplete_runs": sum(1 for r in rows if r.get("tokens_incomplete")),
        "inference_errors": sum(1 for r in rows if r.get("inference_error")),
        "timeouts": sum(1 for r in rows if r.get("timed_out")),
        "invalid_actions": sum(r.get("invalid_actions") or 0 for r in rows),
        "peak_vram_mib": max((r.get("peak_vram_mib") or 0 for r in rows), default=0),
    }


def decide(
    manifest: dict[str, Any], stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The registered rule, with every degenerate case spelled out."""
    spec = manifest["analysis"]
    c, t = stats[spec["control"]], stats[spec["treatment"]]
    if c["successes"] == 0 and t["successes"] == 0:
        return {
            "status": "inconclusive",
            "met": None,
            "reason": "neither arm has a verified success",
        }
    if c["successes"] == 0:
        return {
            "status": "ratio_undefined",
            "met": None,
            "reason": f"control has 0 verified successes (treatment {t['successes']}); "
            "the throughput ratio is undefined, so the rule is not evaluated",
        }
    ratio = t["verified_tasks_per_hour"] / c["verified_tasks_per_hour"]
    met = (
        ratio >= spec["min_ratio"]
        and t["successes"] >= c["successes"]
        and t["inference_errors"] <= c["inference_errors"]
        and t["timeouts"] <= c["timeouts"]
    )
    return {"status": "evaluated", "met": met, "ratio": round(ratio, 2)}


def analyze(manifest: dict[str, Any], runs: str) -> dict[str, Any]:
    state = scan(manifest, runs)
    cells = state["cells"]
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in manifest["arms"]}
    for (arm, seed, task), info in cells.items():
        if info["state"] == "valid":
            rows[arm].append(
                _read_json(
                    os.path.join(cell_dir(runs, arm, seed, task), "results.json")
                )[0]
            )
    counts = {
        name: sum(1 for i in cells.values() if i["state"] == name)
        for name in ("valid", "missing", "invalid")
    }
    stats = {arm: summarize(rows[arm]) for arm in manifest["arms"]}
    complete = counts["valid"] == len(cells) and not state["unexpected"]
    return {
        "experiment": manifest["experiment"],
        "manifest_sha256": manifest_sha256(manifest),
        "matrix": len(cells),
        "counts": counts,
        "unexpected": state["unexpected"],
        "quarantined": state["quarantined"],
        "historical_aborted": manifest.get("aborted", []),
        "complete": complete,
        "arms": stats,
        "decision": decide(manifest, stats)
        if complete
        else {
            "status": "withheld",
            "met": None,
            "reason": "matrix incomplete or invalid",
        },
        "problems": {
            "/".join(map(str, key)): info["problems"]
            for key, info in cells.items()
            if info["problems"]
        },
        "missing": [
            "/".join(map(str, k)) for k, i in cells.items() if i["state"] == "missing"
        ],
    }


def preflight(manifest: dict[str, Any], *, live: bool = False) -> list[str]:
    """Offline consistency of the manifest with the repository (and, with
    `live`, the running Ollama model identity)."""
    problems: list[str] = []
    try:
        pins = _suite_pins(manifest)
    except (OSError, ValueError, KeyError) as exc:
        return [f"suite {manifest['benchmark']['suite']!r} unreadable: {exc}"]
    if not manifest.get("frozen"):
        for arm, spec in manifest["arms"].items():
            if not spec.get("launch"):
                problems.append(
                    f"arm {arm!r} has no launch entry; plain bench.py writes no "
                    "provenance record (the entry must, like exp2_entry does)"
                )
    for task in manifest["tasks"]:
        if task not in pins:
            problems.append(f"task {task!r} is not in suite")
        elif engine.task_digest(task) != pins[task]:
            problems.append(f"task {task!r} fixtures differ from the suite pin")
    if live:
        fingerprint = engine.model_fingerprint(manifest["model"]["tag"])
        for key in ("manifest_digest", "weights_digest"):
            if fingerprint.get(key) != manifest["model"][key]:
                problems.append(f"model {key} differs from the manifest")
    return problems


def _next_free(base: str, mark: str) -> str:
    n = 1
    while os.path.exists(f"{base}{mark}{n}"):
        n += 1
    return f"{base}{mark}{n}"


def promote(
    manifest: dict[str, Any], key: tuple[str, int, str], stage: str, cell: str
) -> list[str]:
    """Validate a finished staging dir, then atomically make it the cell.

    An invalid attempt is quarantined, never deleted and never placed where
    analysis reads it."""
    problems = validate_attempt(manifest, *key, stage)
    if os.path.exists(cell):
        problems.append("cell already exists; refusing to overwrite")
    if problems:
        os.rename(stage, _next_free(cell, ".invalid-"))
        return problems
    os.rename(stage, cell)  # one directory rename: the cell is complete or absent
    return []


def run(manifest: dict[str, Any], runs: str, *, live: bool = True) -> int:
    if manifest.get("frozen"):
        print(f"{manifest['experiment']} is frozen: analysis only.", file=sys.stderr)
        return 2
    problems = preflight(manifest, live=live)
    state = scan(manifest, runs)
    for key, info in state["cells"].items():
        if info["state"] == "invalid":
            problems.append(
                f"{'/'.join(map(str, key))}: " + "; ".join(info["problems"])
            )
    problems += [f"unexpected: {p}" for p in state["unexpected"]]
    if problems:
        print("preflight failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    for arm, seed, task in attempts(manifest):
        cell = cell_dir(runs, arm, seed, task)
        if os.path.exists(os.path.join(cell, "results.json")):
            print(f"SKIP {arm} seed{seed} {task} (valid)")
            continue
        stage = cell + ".partial"
        if os.path.exists(stage):  # a crashed earlier attempt: keep it as evidence
            os.rename(stage, _next_free(cell, ".aborted-"))
        os.makedirs(os.path.dirname(cell), exist_ok=True)
        launch = manifest["arms"][arm].get("launch") or [
            sys.executable,
            os.path.join(ROOT, "bench", "bench.py"),
        ]
        subprocess.run(
            [*launch, "--run-dir", stage, *bench_argv(manifest, seed, task)],
            cwd=ROOT,
            check=False,  # a failed task exits non-zero by design
        )
        if not os.path.isdir(stage):
            print(f"{arm} seed{seed} {task}: no output produced", file=sys.stderr)
            return 3
        with open(os.path.join(stage, "attempt.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "attempt_id": attempt_id(manifest, arm, seed, task),
                    "manifest_sha256": manifest_sha256(manifest),
                },
                f,
            )
        bad = promote(manifest, (arm, seed, task), stage, cell)
        if bad:
            print(f"HALT {arm} seed{seed} {task}: " + "; ".join(bad), file=sys.stderr)
            return 3
        print(f"DONE {arm} seed{seed} {task}")
    return 0


def _print_report(report: dict[str, Any]) -> None:
    c = report["counts"]
    print(
        f"{report['experiment']}: {c['valid']}/{report['matrix']} valid, "
        f"{c['missing']} missing, {c['invalid']} invalid, "
        f"{len(report['unexpected'])} unexpected, "
        f"{len(report['quarantined'])} quarantined, "
        f"{len(report['historical_aborted'])} historical aborted"
    )
    for item in report["historical_aborted"]:
        name = f"{item['arm']} seed{item['seed']} {item['task']}"
        print(f"  aborted {name}: {item['reason']}")
    for key, why in report["problems"].items():
        print(f"  INVALID {key}: {'; '.join(why)}")
    for path in report["unexpected"]:
        print(f"  UNEXPECTED {path}")
    for arm, d in report["arms"].items():
        lo, hi = d["wilson95"]
        rate = (d["success_rate"] or 0) * 100
        print(
            f"\n{arm}: {d['successes']}/{d['attempts']} ({rate:.1f}%, 95% CI "
            f"{lo * 100:.0f}-{hi * 100:.0f}%) {d['verified_tasks_per_hour']} "
            f"tasks/h over {d['total_wall_s']}s"
            + ("  [NO VERIFIED SUCCESS]" if d["outcome"] == "no_success" else "")
        )
    d = report["decision"]
    ratio = f" ratio {d['ratio']}x" if "ratio" in d else ""
    print(f"\ndecision: {d['status']} met={d['met']}{ratio} {d.get('reason', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=("check", "status", "analyze", "plan", "run")
    )
    parser.add_argument("manifest")
    parser.add_argument("--runs", help="records directory (default: manifest runs_dir)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--live", action="store_true", help="also check the Ollama model"
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    runs = os.path.abspath(args.runs or os.path.join(ROOT, manifest["runs_dir"]))

    if args.command == "check":
        problems = preflight(manifest, live=args.live)
        print("\n".join(problems) or "manifest and fixtures consistent")
        return 2 if problems else 0
    if args.command == "plan":
        state = scan(manifest, runs)
        for key, info in state["cells"].items():
            arm, seed, task = key
            print(
                f"{info['state']:8s} {attempt_id(manifest, *key)} {arm} seed{seed} "
                f"{task}"
            )
        return 0
    if args.command == "run":
        return run(manifest, runs)

    report = analyze(manifest, runs)
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        print()
    else:
        _print_report(report)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
