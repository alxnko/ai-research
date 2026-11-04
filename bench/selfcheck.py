#!/usr/bin/env python3
"""Self-check: every task must fail unsolved and pass with its golden solution.

This is the harness's own regression test. Run it after adding or editing a task.
"""

import concurrent.futures
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bench import (
    all_tasks,
    clean_copy,
    install_tests,
    load_task,
    parse_action,
    policy_check,
    score,
    snapshot,
)


def parser_checks():
    """The parser is the agent's only actuator; keep it honest."""
    cases = [
        ("RUN: ls", ("run", "ls")),
        (
            "RUN: cat > a.py << 'EOF'\nx = 1\nEOF",
            ("run", "cat > a.py << 'EOF'\nx = 1\nEOF"),
        ),
        ("RUN: ls\n```", ("run", "ls")),
        ("DONE", ("done", None)),
        ("DONE\nAll fixed.", ("done", None)),
        ("```bash\nls -la\n```", ("run", "ls -la")),
        ("<result>cat webhook.py</result>", ("run", "cat webhook.py")),
        ("<command>ls -la</command>", ("run", "ls -la")),
        (
            "<result>cat > a.py << 'EOF'\nx = 1\nEOF</result>",
            ("run", "cat > a.py << 'EOF'\nx = 1\nEOF"),
        ),
        ("no idea", (None, None)),
        ("", (None, None)),
    ]
    bad = [
        (text, want, parse_action(text))
        for text, want in cases
        if parse_action(text) != want
    ]
    for text, want, got in bad:
        print(f"PARSER {text!r}: want {want}, got {got}")
    return not bad


def attempt(meta, solution):
    """Score through the same path the benchmark uses, including the clean
    workspace rebuild, so the self-check exercises the real scorer."""
    tmp = tempfile.mkdtemp(prefix="benchcheck-")
    work = os.path.join(tmp, "work")
    shutil.copytree(os.path.join(meta["dir"], "repo"), work)
    with open(os.path.join(tmp, "baseline.json"), "w") as f:
        json.dump(snapshot(work), f)
    if solution:
        sol = os.path.join(meta["dir"], "solution")
        for name in os.listdir(sol):
            shutil.copy2(os.path.join(sol, name), os.path.join(work, name))
    passed, out = score(meta, work, tmp)
    diff = policy_check(meta, work, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return (0 if passed else 1), out, diff


def isolation_checks():
    """The scorer must not follow anything the agent can create."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-iso-")
    victim_dir = os.path.join(tmp, "victim")
    work = os.path.join(tmp, "work")
    os.makedirs(victim_dir)
    os.makedirs(work)
    victim = os.path.join(victim_dir, "target.txt")
    with open(victim, "w") as f:
        f.write("ORIGINAL")

    # the agent points a hidden-test path at a host directory
    os.symlink(victim_dir, os.path.join(work, "_hidden"))
    os.symlink(victim, os.path.join(work, "leaked.txt"))
    scoring = os.path.join(tmp, "score")
    clean_copy(work, scoring)
    if os.path.exists(os.path.join(scoring, "_hidden")):
        failures.append("clean_copy kept a symlinked _hidden")
    if os.path.exists(os.path.join(scoring, "leaked.txt")):
        failures.append("clean_copy kept a symlinked file")
    if "leaked.txt" in snapshot(work):
        failures.append("snapshot followed a symlink")
    with open(victim) as f:
        if f.read() != "ORIGINAL":
            failures.append("clean_copy wrote through a symlink")

    # a pre-existing _hidden must stop the installer, not be merged into
    meta = {
        "dir": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tasks", "backend-pagination"
        )
    }
    os.makedirs(os.path.join(scoring, "_hidden"), exist_ok=True)
    try:
        install_tests(meta, scoring)
        failures.append("install_tests merged into an existing _hidden")
    except RuntimeError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


def check_task(tid):
    """Runs in a worker thread; every attempt gets its own temp workspace."""
    meta = load_task(tid)
    base_code, base_out, _ = attempt(meta, solution=False)
    sol_code, sol_out, diff = attempt(meta, solution=True)
    unsolved_ok = base_code != 0
    solved_ok = sol_code == 0 and not diff["forbidden"]
    problems = []
    if not solved_ok:
        problems.append((tid, "solution does not pass", sol_out[-600:]))
    if not unsolved_ok:
        problems.append((tid, "task passes with no work", base_out[-400:]))
    line = (
        f"{'ok' if not problems else 'BROKEN':6s} {tid:24s} "
        f"unsolved_exit={base_code:<4d} solved_exit={sol_code:<4d} "
        f"forbidden={len(diff['forbidden'])}"
    )
    return line, problems


def main():
    failures = []
    if not parser_checks():
        failures.append(("parse_action", "parser regression", ""))
    for problem in isolation_checks():
        failures.append(("isolation", problem, ""))
    ids = all_tasks()
    # Tasks are independent: build workspaces in parallel, report in order.
    workers = min(8, len(ids) or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check_task, ids))
    for line, problems in results:
        print(line)
        failures.extend(problems)

    if failures:
        print("\n--- failures ---")
        for tid, why, out in failures:
            print(f"\n{tid}: {why}\n{out}")
        return 1
    print(f"\n{len(all_tasks())} tasks verified: unsolved fail, solutions pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
