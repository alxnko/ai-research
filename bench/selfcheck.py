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
import bench
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
    """Test heredocs, multiline execution, quotes, and trap handling."""
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

    # Verify exit status is authoritative even if command clears or overrides traps
    trap_cmd = "trap '' EXIT; echo 'trap cleared' >&2; exit 42"
    code, out = run_sandboxed(work, trap_cmd, timeout=15)
    if code != 42 or "trap cleared" not in out:
        errs.append(
            f"exit code or stderr propagation with cleared trap failed ({code}): {out}"
        )

    code, out = run_sandboxed(work, "sleep 5", timeout=1)
    if code != 124 or "timeout after 1s" not in out:
        errs.append(f"command timeout handling failed ({code}): {out}")

    return errs


def _check_sandbox_policy(tmp, work):
    """Test policy check and clean copy against symlinks, dirs, and fifos."""
    errs = []
    leak_target = f"/tmp/bench_host_leak_check_{os.getpid()}"
    if os.path.exists(leak_target):
        os.remove(leak_target)
    run_sandboxed(work, f"touch {leak_target}", timeout=10)
    if os.path.exists(leak_target):
        errs.append("host isolation breached: sandbox wrote to host /tmp")
        os.remove(leak_target)

    # Establish baseline
    base_file = os.path.join(work, "base.txt")
    with open(base_file, "w") as f:
        f.write("base")
    sub_dir = os.path.join(work, "sub_dir")
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "inner.txt"), "w") as f:
        f.write("inner")
    with open(os.path.join(tmp, "baseline.json"), "w") as f:
        json.dump(snapshot(work), f)

    meta = {"allowed_files": ["base.txt", "sub_dir", "sub_dir/inner.txt"]}

    # 1. Symlink to file
    file_symlink = os.path.join(work, "leak_file.lnk")
    os.symlink("/etc/passwd", file_symlink)
    diff = policy_check(meta, work, tmp)
    if "leak_file.lnk" not in diff["forbidden"]:
        errs.append("policy_check failed to detect file symlink as forbidden")

    # 2. Symlink to directory
    dir_symlink = os.path.join(work, "leak_dir.lnk")
    os.symlink("/usr", dir_symlink)
    diff = policy_check(meta, work, tmp)
    if "leak_dir.lnk" not in diff["forbidden"]:
        errs.append("policy_check failed to detect directory symlink as forbidden")

    # 3. Directory permission change
    os.chmod(sub_dir, 0o777)
    diff = policy_check(meta, work, tmp)
    if "sub_dir" not in diff["forbidden"]:
        errs.append("policy_check failed to detect directory permission change")
    os.chmod(sub_dir, 0o755)

    # 4. FIFO detection and clean_copy refusal
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
        if os.path.exists(os.path.join(clean_dst, "leak_file.lnk")):
            errs.append("clean_copy copied a file symlink")
        if os.path.exists(os.path.join(clean_dst, "leak_dir.lnk")):
            errs.append("clean_copy copied a directory symlink")
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


def runner_smoke_test():
    """CI smoke test: executes run_task end-to-end with a mocked model
    in the real bubblewrap sandbox, validating that metadata, run_task,
    scoring, and provenance records operate without runtime regressions."""
    step = 0

    def mock_chat(model, msgs, ctx, ngpu, timeout=1800):
        nonlocal step
        step += 1
        if step == 1:
            return "RUN: cat cache.py", "", 10
        if step == 2:
            sol_path = os.path.join(
                ROOT, "tasks", "debugging-default-arg", "solution", "cache.py"
            )
            with open(sol_path) as sf:
                sol = sf.read()
            fix = f'RUN: cat > cache.py << "EOF"\n{sol}\nEOF\n'
            return fix, "", 25
        return "DONE", "", 5

    tmp = tempfile.mkdtemp(prefix="bench-runner-smoke-")
    failures = []
    try:
        meta = load_task("debugging-default-arg")
        rec = bench.run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=5,
            max_wall=60,
            run_dir=os.path.join(tmp, "run_0"),
            verbose=False,
            chat_fn=mock_chat,
        )
        if not rec["passed"]:
            failures.append(("runner_smoke_test", "mocked run failed", str(rec)))
        if rec["task"] != "debugging-default-arg":
            failures.append(
                ("runner_smoke_test", "task id mismatch in record", str(rec))
            )
        if rec["steps"] != 3:
            failures.append(
                ("runner_smoke_test", f"expected 3 steps, got {rec['steps']}", "")
            )
        if rec["agent_wall_s"] <= 0 or rec["total_wall_s"] <= 0:
            failures.append(
                ("runner_smoke_test", "timing metrics not populated", str(rec))
            )
        if not os.path.exists(os.path.join(tmp, "run_0", "transcript.json")):
            failures.append(("runner_smoke_test", "transcript.json not written", ""))
        if not os.path.exists(os.path.join(tmp, "run_0", "baseline.json")):
            failures.append(("runner_smoke_test", "baseline.json not written", ""))
    except Exception as e:
        failures.append(("runner_smoke_test", f"exception during run_task: {e}", ""))
    finally:
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
    for item in runner_smoke_test():
        failures.append(item)

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
    total = len(all_tasks())
    print(f"\n{total} tasks verified + smoke test: unsolved fail, solutions pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
