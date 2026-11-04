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
    run_sandboxed,
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
    snap = snapshot(work)
    if snap.get("leaked.txt", {}).get("type") != "symlink":
        failures.append("snapshot did not classify symlink correctly")
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


def _check_heredocs_and_quotes(work):
    """Test heredocs, multiline execution, and quotes in sandbox."""
    errs = []
    heredoc_cmd = (
        "cat > hello.py << 'EOF'\n"
        "x = 40 + 2\n"
        "print(f'heredoc_val={x}')\n"
        "EOF\n"
        "python3 hello.py"
    )
    code, out = run_sandboxed(work, heredoc_cmd, timeout=15)
    if code != 0 or "heredoc_val=42" not in out:
        errs.append(f"heredoc execution failed ({code}): {out}")
    elif not os.path.exists(os.path.join(work, "hello.py")):
        errs.append("heredoc did not persist generated file")

    multiline_cmd = "echo 'line 1'\necho 'line 2'"
    code, out = run_sandboxed(work, multiline_cmd, timeout=15)
    if code != 0 or "line 1" not in out or "line 2" not in out:
        errs.append(f"multiline execution failed ({code}): {out}")

    quote_cmd = "echo 'quotes and \"double\" quotes'"
    code, out = run_sandboxed(work, quote_cmd, timeout=15)
    if code != 0 or 'quotes and "double" quotes' not in out:
        errs.append(f"quote execution failed ({code}): {out}")

    code, out = run_sandboxed(work, "echo 'error message' >&2; exit 42", timeout=15)
    if code != 42 or "error message" not in out:
        errs.append(f"exit code or stderr propagation failed ({code}): {out}")

    code, out = run_sandboxed(work, "sleep 5", timeout=1)
    if code != 124 or "timeout after 1s" not in out:
        errs.append(f"command timeout handling failed ({code}): {out}")

    return errs


def _check_sandbox_policy(tmp, work):
    """Test policy check and clean copy against symlinks and fifos."""
    errs = []
    leak_target = f"/tmp/bench_host_leak_check_{os.getpid()}"
    if os.path.exists(leak_target):
        os.remove(leak_target)
    run_sandboxed(work, f"touch {leak_target}", timeout=10)
    if os.path.exists(leak_target):
        errs.append("host isolation breached: sandbox wrote to host /tmp")
        os.remove(leak_target)

    with open(os.path.join(work, "base.txt"), "w") as f:
        f.write("base")
    with open(os.path.join(tmp, "baseline.json"), "w") as f:
        json.dump(snapshot(work), f)

    meta = {"allowed_files": ["base.txt"]}
    os.symlink("/etc/passwd", os.path.join(work, "leak.lnk"))
    diff = policy_check(meta, work, tmp)
    if "leak.lnk" not in diff["forbidden"]:
        errs.append("policy_check failed to detect symlink as forbidden")

    fifo_path = os.path.join(work, "test_fifo")
    try:
        os.mkfifo(fifo_path)
        diff = policy_check(meta, work, tmp)
        if "test_fifo" not in diff["forbidden"]:
            errs.append("policy_check failed to detect FIFO as forbidden")
        clean_dst = os.path.join(tmp, "clean_test")
        clean_copy(work, clean_dst)
        if os.path.exists(os.path.join(clean_dst, "test_fifo")):
            errs.append("clean_copy copied a FIFO")
        if os.path.exists(os.path.join(clean_dst, "leak.lnk")):
            errs.append("clean_copy copied a symlink")
    except OSError:
        pass
    return errs


def sandbox_regression_checks():
    """Exercise sandboxed execution directly against real shell edge cases."""
    tmp = tempfile.mkdtemp(prefix="bench-sandbox-reg-")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    failures = []
    failures.extend(_check_heredocs_and_quotes(work))
    failures.extend(_check_sandbox_policy(tmp, work))
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
    for problem in sandbox_regression_checks():
        failures.append(("sandbox_regression", problem, ""))
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
