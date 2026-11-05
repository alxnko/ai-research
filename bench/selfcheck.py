#!/usr/bin/env python3
"""Self-check verification suite for benchmark harness, tasks, and isolation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench.bench import (
    _mock_chat_stream,
    all_tasks,
    check_cgroup_v2,
    clean_copy,
    install_tests,
    load_task,
    parse_action,
    policy_check,
    prepare,
    run_sandboxed,
    run_task,
    score,
    snapshot,
)
from bench.classifier import classify_task_run
from bench.telemetry import (
    build_telemetry_record,
    parse_stream_chunk_event,
)


def parser_checks() -> list[tuple[str, str, str]]:
    """The parser is the agent's only actuator in text mode; keep it honest."""
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
    failures = []
    for text, want in cases:
        got = parse_action(text)
        if got != want:
            failures.append(
                ("parser_checks", f"text={text!r}", f"want={want} got={got}")
            )
    return failures


def streaming_parser_unit_tests() -> list[tuple[str, str, str]]:
    """Unit test streaming event parsing, TTFT timestamping, and duration metrics."""
    failures = []
    state: dict[str, Any] = {
        "content": "",
        "thinking": "",
        "tool_calls": [],
        "done": False,
        "ttft_any_s": None,
        "ttft_thinking_s": None,
        "ttft_content_s": None,
        "ttft_tool_s": None,
        "done_reason": "stop",
        "prompt_eval_count": 0,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration_ns": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
    }
    t0 = time.perf_counter()

    # 1. Thinking chunk
    parse_stream_chunk_event(
        {"message": {"thinking": "analyzing..."}},
        t0,
        state,
    )
    if state["thinking"] != "analyzing...":
        failures.append(
            ("streaming_parser", "thinking text mismatch", state["thinking"])
        )
    if state["ttft_thinking_s"] is None or state["ttft_any_s"] is None:
        failures.append(("streaming_parser", "ttft_thinking not recorded", str(state)))

    # 2. Content chunk
    parse_stream_chunk_event(
        {"message": {"content": "RUN: ls"}},
        t0,
        state,
    )
    if state["content"] != "RUN: ls":
        failures.append(("streaming_parser", "content text mismatch", state["content"]))
    if state["ttft_content_s"] is None:
        failures.append(("streaming_parser", "ttft_content not recorded", str(state)))

    # 3. Tool call chunk
    tool_chunk = {
        "message": {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "execute_command",
                        "arguments": {"command": "ls"},
                    },
                }
            ]
        }
    }
    parse_stream_chunk_event(tool_chunk, t0, state)
    if len(state["tool_calls"]) != 1:
        failures.append(
            ("streaming_parser", "tool call not accumulated", str(state["tool_calls"]))
        )
    if state["ttft_tool_s"] is None:
        failures.append(("streaming_parser", "ttft_tool not recorded", str(state)))

    # 4. Final done chunk
    done_chunk = {
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 50,
        "prompt_eval_cached_count": 20,
        "prompt_eval_duration": 500_000_000,
        "eval_count": 25,
        "eval_duration": 250_000_000,
        "total_duration": 800_000_000,
        "load_duration": 50_000_000,
    }
    parse_stream_chunk_event(done_chunk, t0, state)
    if not state["done"]:
        failures.append(("streaming_parser", "done flag not set", str(state)))

    rec = build_telemetry_record(state, wall_duration_s=0.85, step=1, request_idx=1)
    if rec["prompt_eval_count"] != 50 or rec["eval_count"] != 25:
        failures.append(("streaming_parser", "token count mismatch", str(rec)))
    if rec["decode_tokens_per_s"] != 100.0:  # 25 / 0.25s = 100
        failures.append(("streaming_parser", "decode_tps mismatch", str(rec)))
    if rec["prompt_eval_tokens_per_s"] != 100.0:  # 50 / 0.5s = 100
        failures.append(("streaming_parser", "prompt_eval_tps mismatch", str(rec)))

    return failures


def classifier_unit_tests() -> list[tuple[str, str, str]]:
    """Unit test failure classification outcome hierarchy and empirical cause rules."""
    failures = []

    # 1. Passed task has no failure cause
    c_pass = classify_task_run(
        passed=True,
        timed_out=False,
        step_count=3,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="OK",
        steps=[],
    )
    if c_pass["outcome"] != "passed" or c_pass["possible_cause"] is not None:
        failures.append(("classifier", "passed classification failed", str(c_pass)))

    # 2. OOM / memory kill -> memory_failure + resource_constraint
    c_oom = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=2,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=[],
        exit_status="Killed (cgroup: oom-kill)",
        steps=[],
    )
    if (
        c_oom["outcome"] != "memory_failure"
        or c_oom["possible_cause"] != "resource_constraint"
    ):
        failures.append(("classifier", "oom classification failed", str(c_oom)))

    # 3. Forbidden file modified -> forbidden_file_change
    c_forbid = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=5,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=["test_runner.py"],
        changed_files=["test_runner.py", "app.py"],
        exit_status="FAIL",
        steps=[],
    )
    if c_forbid["outcome"] != "forbidden_file_change":
        failures.append(
            ("classifier", "forbidden files classification failed", str(c_forbid))
        )

    # 4. Repeated command loop -> repeated_command
    loop_steps = [
        {"command": "python test.py", "output": "SyntaxError"},
        {"command": "python test.py", "output": "SyntaxError"},
        {"command": "python test.py", "output": "SyntaxError"},
    ]
    c_loop = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=3,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=[],
        exit_status="FAIL",
        steps=loop_steps,
    )
    if c_loop["outcome"] != "repeated_command":
        failures.append(("classifier", "loop classification failed", str(c_loop)))

    # 5. Invalid action abort -> invalid_action + tool_format_incompatibility
    c_invalid = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=3,
        max_steps=20,
        invalid_actions=3,
        forbidden_files=[],
        changed_files=[],
        exit_status="none",
        steps=[],
    )
    if (
        c_invalid["outcome"] != "invalid_action"
        or c_invalid["possible_cause"] != "tool_format_incompatibility"
    ):
        failures.append(
            ("classifier", "invalid action classification failed", str(c_invalid))
        )

    # 6. Context exhaustion -> context_loss
    c_ctx = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=10,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="FAIL",
        steps=[],
        num_ctx=4096,
        telemetry_records=[{"prompt_eval_count": 4096}],
    )
    if c_ctx["possible_cause"] != "context_loss":
        failures.append(
            ("classifier", "context_loss classification failed", str(c_ctx))
        )

    # 7. Insufficient reasoning: within context budget (< 85% num_ctx)
    # but verification failed
    c_reason = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=10,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="FAIL",
        steps=[],
        num_ctx=32768,
        telemetry_records=[{"prompt_eval_count": 1200}],
    )
    if c_reason["possible_cause"] != "insufficient_reasoning":
        failures.append(
            (
                "classifier",
                "insufficient_reasoning classification failed",
                str(c_reason),
            )
        )

    # 8. Unsubstantiated causes must remain unknown
    c_unk = classify_task_run(
        passed=False,
        timed_out=False,
        step_count=10,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="FAIL",
        steps=[],
        num_ctx=32768,
        telemetry_records=[],  # no telemetry evidence
    )
    if c_unk["possible_cause"] != "unknown":
        failures.append(("classifier", "unknown cause default failed", str(c_unk)))

    return failures


def strict_isolation_test() -> list[tuple[str, str, str]]:
    """Verify --strict-isolation flag fails closed when cgroups are unavailable."""
    failures = []
    cgroup_available = check_cgroup_v2()

    # When cgroup is available, --strict-isolation should pass initial validation
    env = os.environ.copy()
    env["BENCH_MOCK_MODEL"] = "1"
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "bench", "bench.py"),
        "--strict-isolation",
        "--task",
        "debugging-default-arg",
        "--tag",
        "test-strict-iso",
    ]
    tmp_out = tempfile.mkdtemp(prefix="bench-strict-test-")
    try:
        proc = subprocess.run(
            [*cmd, "--run-dir", tmp_out],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cgroup_available and proc.returncode != 0:
            failures.append(
                (
                    "strict_isolation",
                    "failed with strict-isolation when cgroups are available",
                    proc.stderr,
                )
            )
        elif not cgroup_available and proc.returncode == 0:
            failures.append(
                (
                    "strict_isolation",
                    "did not fail closed when cgroups were unavailable",
                    proc.stdout,
                )
            )
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

    return failures


def subprocess_resume_integration_test() -> list[tuple[str, str, str]]:
    """Genuine subprocess CLI test for interrupted run resumption:
    1. Run task 1 via CLI.
    2. Add corrupted workspace for task 2.
    3. Resume via CLI and verify task 2 is rebuilt and executed cleanly.
    4. Verify both tasks in results.json.
    5. Verify mismatched model resume fails closed.
    """
    tmp_run = tempfile.mkdtemp(prefix="bench-subproc-resume-")
    failures = []
    bench_py = os.path.join(REPO_ROOT, "bench", "bench.py")
    env = os.environ.copy()
    env["BENCH_MOCK_MODEL"] = "1"

    try:
        # Step 1: Run single task
        cmd1 = [
            sys.executable,
            bench_py,
            "--run-dir",
            tmp_run,
            "--task",
            "debugging-default-arg",
            "--model",
            "qwen2.5-coder:7b",
        ]
        res1 = subprocess.run(cmd1, env=env, capture_output=True, text=True, timeout=60)
        if res1.returncode != 0:
            failures.append(
                (
                    "subprocess_resume",
                    "initial run failed",
                    f"{res1.stdout}\n{res1.stderr}",
                )
            )
            return failures

        # Step 2: Inject dirty workspace for second task
        task2_id = "debugging-off-by-one"
        dirty_dir = os.path.join(tmp_run, "r0", task2_id, "work")
        os.makedirs(dirty_dir, exist_ok=True)
        dirty_file = os.path.join(dirty_dir, "corrupt.txt")
        with open(dirty_file, "w") as f:
            f.write("corrupted workspace from interrupted run")

        # Step 3: Resume run with task2
        cmd2 = [
            sys.executable,
            bench_py,
            "--resume",
            tmp_run,
            "--task",
            task2_id,
        ]
        res2 = subprocess.run(cmd2, env=env, capture_output=True, text=True, timeout=60)
        if res2.returncode != 0:
            failures.append(
                (
                    "subprocess_resume",
                    "resume run failed",
                    f"{res2.stdout}\n{res2.stderr}",
                )
            )

        # Step 4: Verify dirty state was purged and results.json has both tasks
        results_file = os.path.join(tmp_run, "results.json")
        if not os.path.exists(results_file):
            failures.append(("subprocess_resume", "results.json not found", ""))
            return failures

        with open(results_file) as f:
            recs = json.load(f)
        tasks_in_res = [r["task"] for r in recs]
        if "debugging-default-arg" not in tasks_in_res:
            failures.append(
                ("subprocess_resume", "task 1 missing from results", str(tasks_in_res))
            )
        if task2_id not in tasks_in_res:
            failures.append(
                ("subprocess_resume", "task 2 missing from results", str(tasks_in_res))
            )

        if os.path.exists(dirty_file):
            failures.append(
                (
                    "subprocess_resume",
                    "dirty workspace file not cleaned on resume",
                    dirty_file,
                )
            )

        # Step 5: Test model mismatch rejection on resume
        cmd_mismatch = [
            sys.executable,
            bench_py,
            "--resume",
            tmp_run,
            "--model",
            "different-model:latest",
        ]
        res_mis = subprocess.run(
            cmd_mismatch, env=env, capture_output=True, text=True, timeout=10
        )
        if res_mis.returncode == 0:
            failures.append(
                (
                    "subprocess_resume",
                    "resume allowed mismatched model without failing",
                    "",
                )
            )
        elif "model mismatch" not in res_mis.stderr:
            failures.append(
                (
                    "subprocess_resume",
                    "model mismatch error message missing",
                    res_mis.stderr,
                )
            )

    finally:
        shutil.rmtree(tmp_run, ignore_errors=True)

    return failures


def runner_actuator_smoke_tests() -> list[tuple[str, str, str]]:
    """Smoke test text and native_tools actuator paths end-to-end with mock model."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-actuator-smoke-")
    meta = load_task("debugging-default-arg")
    try:
        # 1. Text actuator
        r_text = os.path.join(tmp, "text_arm")
        rec_text = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=5,
            max_wall=60,
            run_dir=r_text,
            actuator="text",
            verbose=False,
            chat_fn=_mock_chat_stream,
        )
        if not rec_text["passed"]:
            failures.append(
                ("actuator_smoke", "text actuator run failed", str(rec_text))
            )
        if rec_text["actuator"] != "text":
            failures.append(
                ("actuator_smoke", "text actuator arm mismatch", str(rec_text))
            )

        # 2. Native tools actuator
        r_tools = os.path.join(tmp, "tools_arm")
        rec_tools = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=5,
            max_wall=60,
            run_dir=r_tools,
            actuator="native_tools",
            verbose=False,
            chat_fn=_mock_chat_stream,
        )
        if not rec_tools["passed"]:
            failures.append(
                ("actuator_smoke", "native_tools actuator run failed", str(rec_tools))
            )
        if rec_tools["actuator"] != "native_tools":
            failures.append(
                ("actuator_smoke", "native_tools arm mismatch", str(rec_tools))
            )
        if rec_tools["tool_calls"] < 1:
            failures.append(
                ("actuator_smoke", "no tool calls recorded", str(rec_tools))
            )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return failures


def _check_heredocs_and_quotes(work: str) -> list[str]:
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

    trap_cmd = "trap '' EXIT; echo 'trap cleared' >&2; exit 42"
    code, out = run_sandboxed(work, trap_cmd, timeout=15)
    if code != 42 or "trap cleared" not in out:
        errs.append(
            f"exit code or stderr propagation with cleared trap failed ({code}): {out}"
        )

    code, out = run_sandboxed(work, "sleep 5", timeout=1)
    if code != -1 or "timed out after 1s" not in out:
        errs.append(f"command timeout handling failed ({code}): {out}")

    return errs


def isolation_checks() -> list[tuple[str, str, str]]:
    """The scorer must not follow anything the agent can create."""
    failures: list[tuple[str, str, str]] = []
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
        failures.append(("isolation", "clean_copy kept a symlinked _hidden", ""))
    if os.path.exists(os.path.join(scoring, "leaked.txt")):
        failures.append(("isolation", "clean_copy kept a symlinked file", ""))
    snap = snapshot(work)
    if "leaked.txt" in snap:
        failures.append(("isolation", "snapshot followed a symlink", ""))
    with open(victim) as f:
        if f.read() != "ORIGINAL":
            failures.append(("isolation", "clean_copy wrote through a symlink", ""))

    # a pre-existing _hidden must stop the installer, not be merged into
    meta = {
        "dir": os.path.join(REPO_ROOT, "bench", "tasks", "backend-pagination"),
        "task_dir": os.path.join(REPO_ROOT, "bench", "tasks", "backend-pagination"),
    }
    os.makedirs(os.path.join(scoring, "_hidden"), exist_ok=True)
    try:
        install_tests(meta, scoring)
        failures.append(
            ("isolation", "install_tests merged into an existing _hidden", "")
        )
    except RuntimeError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


def sandbox_regression_checks() -> list[tuple[str, str, str]]:
    """Exercise sandboxed execution directly against real shell edge cases."""
    tmp = tempfile.mkdtemp(prefix="bench-sandbox-reg-")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    failures = []
    for err in _check_heredocs_and_quotes(work):
        failures.append(("sandbox_regression", err, ""))
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


def attempt(
    meta: dict[str, Any], solution: bool
) -> tuple[int, str, dict[str, list[str]]]:
    """Score through the same path the benchmark uses."""
    tmp = tempfile.mkdtemp(prefix="benchcheck-")
    work = prepare(meta, tmp)
    if solution:
        sol = meta["solution_dir"]
        for name in os.listdir(sol):
            src = os.path.join(sol, name)
            dst = os.path.join(work, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    passed, out = score(meta, work, tmp)
    diff = policy_check(meta, work, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return (0 if passed else 1), out, diff


def check_task(tid: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Runs in a worker thread; every attempt gets its own temp workspace."""
    meta = load_task(tid)
    base_code, base_out, _ = attempt(meta, solution=False)
    sol_code, sol_out, diff = attempt(meta, solution=True)
    unsolved_ok = base_code != 0
    solved_ok = sol_code == 0 and not diff["forbidden"]
    problems: list[tuple[str, str, str]] = []
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


def main() -> int:
    failures: list[tuple[str, str, str]] = []

    print("Running parser checks...")
    failures.extend(parser_checks())

    print("Running streaming parser unit tests...")
    failures.extend(streaming_parser_unit_tests())

    print("Running classifier unit tests...")
    failures.extend(classifier_unit_tests())

    print("Running strict isolation checks...")
    failures.extend(strict_isolation_test())

    print("Running isolation attack checks...")
    failures.extend(isolation_checks())

    print("Running subprocess resume integration tests...")
    failures.extend(subprocess_resume_integration_test())

    print("Running actuator smoke tests (text and native_tools)...")
    failures.extend(runner_actuator_smoke_tests())

    print("Running sandbox regression checks...")
    failures.extend(sandbox_regression_checks())

    ids = all_tasks()
    print(f"Running golden solution checks for {len(ids)} tasks in parallel...")
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
    print(f"\n{total} tasks verified + all regression/integration tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
