#!/usr/bin/env python3
"""Self-check: every task must fail unsolved and pass with its golden solution.

This is the harness's own regression test. Run it after adding or editing a task.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bench import all_tasks, load_task, run_sandboxed  # noqa: E402
from bench import parse_action  # noqa: E402


def parser_checks():
    """The parser is the agent's only actuator; keep it honest."""
    cases = [
        ("RUN: ls", ("run", "ls")),
        ("RUN: cat > a.py << 'EOF'\nx = 1\nEOF", ("run", "cat > a.py << 'EOF'\nx = 1\nEOF")),
        ("RUN: ls\n```", ("run", "ls")),
        ("DONE", ("done", None)),
        ("DONE\nAll fixed.", ("done", None)),
        ("```bash\nls -la\n```", ("run", "ls -la")),
        ("<result>cat webhook.py</result>", ("run", "cat webhook.py")),
        ("<command>ls -la</command>", ("run", "ls -la")),
        ("<result>cat > a.py << 'EOF'\nx = 1\nEOF</result>",
         ("run", "cat > a.py << 'EOF'\nx = 1\nEOF")),
        ("no idea", (None, None)),
        ("", (None, None)),
    ]
    bad = [(text, want, parse_action(text)) for text, want in cases
           if parse_action(text) != want]
    for text, want, got in bad:
        print(f"PARSER {text!r}: want {want}, got {got}")
    return not bad


def attempt(meta, solution):
    tmp = tempfile.mkdtemp(prefix="benchcheck-")
    work = os.path.join(tmp, "work")
    shutil.copytree(os.path.join(meta["dir"], "repo"), work)
    if solution:
        sol = os.path.join(meta["dir"], "solution")
        for name in os.listdir(sol):
            shutil.copy2(os.path.join(sol, name), os.path.join(work, name))
    shutil.copytree(os.path.join(meta["dir"], "tests"),
                    os.path.join(work, "_hidden"))
    open(os.path.join(work, "_hidden", "__init__.py"), "a").close()
    code, out = run_sandboxed(work, meta["test_cmd"], meta.get("timeout_s", 120))
    shutil.rmtree(tmp, ignore_errors=True)
    return code, out


def main():
    failures = []
    if not parser_checks():
        failures.append(("parse_action", "parser regression", ""))
    for tid in all_tasks():
        meta = load_task(tid)
        base_code, base_out = attempt(meta, solution=False)
        sol_code, sol_out = attempt(meta, solution=True)
        unsolved_ok = base_code != 0
        solved_ok = sol_code == 0
        status = "ok" if (unsolved_ok and solved_ok) else "BROKEN"
        print(f"{status:6s} {tid:24s} unsolved_exit={base_code:<4d} "
              f"solved_exit={sol_code}")
        if not solved_ok:
            failures.append((tid, "solution does not pass", sol_out[-600:]))
        if not unsolved_ok:
            failures.append((tid, "task passes with no work", base_out[-400:]))

    if failures:
        print("\n--- failures ---")
        for tid, why, out in failures:
            print(f"\n{tid}: {why}\n{out}")
        return 1
    print(f"\n{len(all_tasks())} tasks verified: unsolved fail, solutions pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
