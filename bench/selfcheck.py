#!/usr/bin/env python3
"""Self-check verification suite for benchmark harness, tasks, and isolation."""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import multiprocessing as mp
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench import bench as bench_mod
from bench.bench import (
    _mock_chat_stream,
    all_tasks,
    assert_sandbox_python,
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
    StreamResult,
    build_telemetry_record,
    chat_stream,
    parse_stream_chunk_event,
    summarize_inference,
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
        "done_reason": None,
        "prompt_eval_count": 0,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration_ns": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
    }
    t0 = time.perf_counter()

    parse_stream_chunk_event({"message": {"thinking": "analyzing..."}}, t0, state)
    if state["thinking"] != "analyzing...":
        failures.append(
            ("streaming_parser", "thinking text mismatch", state["thinking"])
        )
    if state["ttft_thinking_s"] is None or state["ttft_any_s"] is None:
        failures.append(("streaming_parser", "ttft_thinking not recorded", str(state)))

    parse_stream_chunk_event({"message": {"content": "RUN: ls"}}, t0, state)
    if state["content"] != "RUN: ls":
        failures.append(("streaming_parser", "content text mismatch", state["content"]))
    if state["ttft_content_s"] is None:
        failures.append(("streaming_parser", "ttft_content not recorded", str(state)))

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
    if rec["incomplete_stream"]:
        failures.append(
            ("streaming_parser", "completed stream marked as incomplete", str(rec))
        )

    infer_summary = summarize_inference([rec])
    if "token_weighted_decode_tokens_per_s" not in infer_summary:
        failures.append(
            (
                "streaming_parser",
                "token_weighted_decode_tokens_per_s missing",
                str(infer_summary),
            )
        )
    return failures


def premature_stream_closure_unit_tests() -> list[tuple[str, str, str]]:
    """Test premature stream truncation detection and execution rejection."""
    failures = []
    base_state: dict[str, Any] = {
        "content": "",
        "thinking": "",
        "tool_calls": [],
        "done": False,
        "ttft_any_s": 0.05,
        "ttft_thinking_s": None,
        "ttft_content_s": None,
        "ttft_tool_s": None,
        "done_reason": None,
        "prompt_eval_count": 10,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration_ns": 100_000_000,
        "eval_count": 5,
        "eval_duration_ns": 50_000_000,
        "total_duration_ns": 150_000_000,
        "load_duration_ns": 10_000_000,
    }

    # Case A: truncated after thinking
    st_a = dict(base_state, thinking="partial thought...", ttft_thinking_s=0.05)
    rec_a = build_telemetry_record(st_a, wall_duration_s=0.2, step=1, request_idx=1)
    if not rec_a["incomplete_stream"] or rec_a["done_reason"] != (
        "stream_truncated_without_done"
    ):
        failures.append(
            ("stream_truncation", "thinking truncation not flagged", str(rec_a))
        )

    # Case B: truncated after content
    st_b = dict(base_state, content="RUN: echo 1", ttft_content_s=0.08)
    rec_b = build_telemetry_record(st_b, wall_duration_s=0.3, step=1, request_idx=1)
    if not rec_b["incomplete_stream"] or rec_b["done_reason"] != (
        "stream_truncated_without_done"
    ):
        failures.append(
            ("stream_truncation", "content truncation not flagged", str(rec_b))
        )

    # Case C: truncated after tool call
    st_c = dict(
        base_state,
        tool_calls=[{"id": "c1", "function": {"name": "execute_command"}}],
        ttft_tool_s=0.09,
    )
    rec_c = build_telemetry_record(st_c, wall_duration_s=0.4, step=1, request_idx=1)
    if not rec_c["incomplete_stream"] or rec_c["done_reason"] != (
        "stream_truncated_without_done"
    ):
        failures.append(
            ("stream_truncation", "tool call truncation not flagged", str(rec_c))
        )

    # Case D: End-to-end incomplete stream rejection in run_task
    tmp = tempfile.mkdtemp(prefix="bench-incomplete-stream-")
    try:
        meta = load_task("debugging-default-arg")

        def mock_truncated_stream(*args: Any, **kwargs: Any) -> StreamResult:
            return StreamResult(
                content="",
                thinking="thinking truncated...",
                tool_calls=[
                    {
                        "id": "c_trunc",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": "touch should_never_exist.txt"},
                        },
                    }
                ],
                telemetry={
                    "incomplete_stream": True,
                    "done_reason": "stream_truncated_without_done",
                    "done": False,
                    "eval_count": 5,
                    "prompt_eval_count": 10,
                },
            )

        run_res = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=3,
            max_wall=30,
            run_dir=os.path.join(tmp, "trunc_run"),
            actuator="native_tools",
            verbose=False,
            chat_fn=mock_truncated_stream,
        )

        if run_res["outcome"] != "inference_failure":
            failures.append(
                (
                    "stream_truncation",
                    f"expected outcome 'inference_failure', got {run_res['outcome']}",
                    str(run_res),
                )
            )
        if not run_res.get("symptoms", {}).get("incomplete_stream"):
            failures.append(
                (
                    "stream_truncation",
                    "symptom 'incomplete_stream' was not True",
                    str(run_res),
                )
            )
        if os.path.exists(
            os.path.join(tmp, "trunc_run", "work", "should_never_exist.txt")
        ):
            failures.append(
                (
                    "stream_truncation",
                    "partial command from truncated stream was executed!",
                    "",
                )
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return failures


def retry_stream_and_scoring_integrity_tests() -> list[tuple[str, str, str]]:
    """Test retry stream rejection and verification_passed vs run_successful scoring."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-retry-stream-")
    meta = load_task("debugging-default-arg")
    try:
        # 1. Retry stream ends prematurely -> no command executed, inference failure
        call_count = 0

        def mock_retry_truncated(*args: Any, **kwargs: Any) -> StreamResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StreamResult(
                    content="",
                    thinking="I should run a command to touch retry_never.txt",
                    tool_calls=[],
                    telemetry={
                        "eval_count": 10,
                        "prompt_eval_count": 10,
                        "done": True,
                        "done_reason": "stop",
                    },
                )
            return StreamResult(
                content="RUN: touch retry_should_not_run.txt",
                thinking="",
                tool_calls=[],
                telemetry={
                    "incomplete_stream": True,
                    "done_reason": "stream_truncated_without_done",
                    "done": False,
                    "eval_count": 5,
                    "prompt_eval_count": 20,
                },
            )

        run_e = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=3,
            max_wall=30,
            run_dir=os.path.join(tmp, "retry_trunc"),
            actuator="text",
            verbose=False,
            chat_fn=mock_retry_truncated,
        )

        if run_e["outcome"] != "inference_failure":
            failures.append(
                (
                    "retry_truncation",
                    f"expected outcome 'inference_failure', got {run_e['outcome']}",
                    str(run_e),
                )
            )
        if run_e.get("run_successful") or run_e.get("passed"):
            failures.append(
                (
                    "retry_truncation",
                    "run_successful or passed should be False on incomplete retry",
                    str(run_e),
                )
            )
        if os.path.exists(
            os.path.join(tmp, "retry_trunc", "work", "retry_should_not_run.txt")
        ):
            failures.append(
                (
                    "retry_truncation",
                    "command from truncated retry stream was executed!",
                    "",
                )
            )

        # 2. Valid patch followed by failed inference
        # Step 1 applies solution; step 2 fails with incomplete stream.
        # verification_passed must be True, but run_successful and passed False.
        with open(os.path.join(meta["solution_dir"], "cache.py")) as f:
            cache_code = f.read()

        step_idx = 0

        def mock_solve_then_fail(*args: Any, **kwargs: Any) -> StreamResult:
            nonlocal step_idx
            step_idx += 1
            if step_idx == 1:
                return StreamResult(
                    content=f"RUN: cat << 'EOF' > cache.py\n{cache_code}\nEOF",
                    thinking="applying fix",
                    tool_calls=[],
                    telemetry={
                        "eval_count": 10,
                        "prompt_eval_count": 10,
                        "done": True,
                        "done_reason": "stop",
                    },
                )
            return StreamResult(
                content="",
                thinking="thinking...",
                tool_calls=[],
                telemetry={
                    "incomplete_stream": True,
                    "done_reason": "stream_truncated_without_done",
                    "done": False,
                    "eval_count": 5,
                    "prompt_eval_count": 10,
                },
            )

        run_f = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=4,
            max_wall=30,
            run_dir=os.path.join(tmp, "solve_then_fail"),
            actuator="text",
            verbose=False,
            chat_fn=mock_solve_then_fail,
        )

        if not run_f.get("verification_passed"):
            failures.append(
                (
                    "scoring_integrity",
                    "verification_passed should be True because patch solved tests",
                    str(run_f),
                )
            )
        if run_f.get("run_successful") or run_f.get("passed"):
            failures.append(
                (
                    "scoring_integrity",
                    "run_successful and passed must be False due to inference error",
                    str(run_f),
                )
            )
        if run_f["outcome"] != "inference_failure":
            failures.append(
                (
                    "scoring_integrity",
                    f"expected outcome 'inference_failure', got {run_f['outcome']}",
                    str(run_f),
                )
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return failures


class _MockStreamServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
    mode: str = "normal"


class _MockStreamHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        mode = getattr(self.server, "mode", "normal")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        if mode == "infinite_stream":
            i = 0
            try:
                while True:
                    chunk = {
                        "message": {"content": f"tok_{i} "},
                        "done": False,
                    }
                    self.wfile.write((json.dumps(chunk) + "\n").encode())
                    self.wfile.flush()
                    i += 1
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        elif mode == "stall_after_initial":
            chunk = {"message": {"content": "initial data "}, "done": False}
            self.wfile.write((json.dumps(chunk) + "\n").encode())
            self.wfile.flush()
            time.sleep(5.0)
        elif mode == "delay_first":
            time.sleep(5.0)
            try:
                chunk = {"message": {"content": "too late"}, "done": True}
                self.wfile.write((json.dumps(chunk) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        elif mode == "incomplete_tool":
            chunk = {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c_timeout",
                            "function": {
                                "name": "execute_command",
                                "arguments": {
                                    "command": "touch should_never_exist_timeout.txt"
                                },
                            },
                        }
                    ]
                },
                "done": False,
            }
            self.wfile.write((json.dumps(chunk) + "\n").encode())
            self.wfile.flush()
            time.sleep(5.0)
        elif mode == "malformed_chunk":
            self.wfile.write(b"not a valid json chunk\n")
            self.wfile.flush()
        else:
            chunk1 = {"message": {"content": "Hello"}, "done": False}
            chunk2 = {
                "message": {"content": " world"},
                "done": True,
                "eval_count": 2,
                "prompt_eval_count": 10,
            }
            self.wfile.write((json.dumps(chunk1) + "\n").encode())
            self.wfile.write((json.dumps(chunk2) + "\n").encode())
            self.wfile.flush()

    def log_message(self, *args: Any) -> None:
        pass


def inference_deadline_regression_tests() -> list[tuple[str, str, str]]:
    """Test cancellable worker, monotonic deadline enforcement, and absence of lingering activity."""
    failures: list[tuple[str, str, str]] = []
    server = _MockStreamServer(("127.0.0.1", 0), _MockStreamHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"

    try:
        # 1. Continuous token streaming beyond deadline
        server.mode = "infinite_stream"
        t0 = time.monotonic()
        res1 = chat_stream("mock", [{"role": "user", "content": "hi"}], 2048, 0, timeout=0.25, url=url)
        dur1 = time.monotonic() - t0
        if not res1.telemetry.get("timed_out"):
            failures.append(("inference_deadline", "infinite stream: timed_out not True", str(res1.telemetry)))
        if res1.telemetry.get("done_reason") != "deadline_exceeded":
            failures.append(("inference_deadline", "infinite stream: done_reason != deadline_exceeded", str(res1.telemetry)))
        if not res1.telemetry.get("tokens_incomplete"):
            failures.append(("inference_deadline", "infinite stream: tokens_incomplete not True", str(res1.telemetry)))
        if not res1.content:
            failures.append(("inference_deadline", "infinite stream: partial content not recorded", str(res1)))
        if dur1 > 0.75:
            failures.append(("inference_deadline", "infinite stream: deadline not enforced in time", f"dur={dur1}s"))

        # 2. Connection that stops sending data after initial output
        server.mode = "stall_after_initial"
        t0 = time.monotonic()
        res2 = chat_stream("mock", [{"role": "user", "content": "hi"}], 2048, 0, timeout=0.25, url=url)
        dur2 = time.monotonic() - t0
        if not res2.telemetry.get("timed_out"):
            failures.append(("inference_deadline", "stall after initial: timed_out not True", str(res2.telemetry)))
        if not res2.telemetry.get("tokens_incomplete"):
            failures.append(("inference_deadline", "stall after initial: tokens_incomplete not True", str(res2.telemetry)))
        if res2.content != "initial data ":
            failures.append(("inference_deadline", "stall after initial: initial content missing", res2.content))
        if dur2 > 0.75:
            failures.append(("inference_deadline", "stall after initial: took too long", f"dur={dur2}s"))

        # 3. Server that delays its first response
        server.mode = "delay_first"
        t0 = time.monotonic()
        res3 = chat_stream("mock", [{"role": "user", "content": "hi"}], 2048, 0, timeout=0.25, url=url)
        dur3 = time.monotonic() - t0
        if not res3.telemetry.get("timed_out"):
            failures.append(("inference_deadline", "delayed first: timed_out not True", str(res3.telemetry)))
        if dur3 > 0.75:
            failures.append(("inference_deadline", "delayed first: took too long", f"dur={dur3}s"))

        # 4. Incomplete tool call at deadline: ensures no command is executed
        server.mode = "incomplete_tool"
        tmp = tempfile.mkdtemp(prefix="bench-tool-timeout-")
        try:
            meta = load_task("debugging-default-arg")

            def custom_chat_fn(*args: Any, **kwargs: Any) -> StreamResult:
                kwargs["url"] = url
                return chat_stream(*args, **kwargs)

            run_res = run_task(
                meta=meta,
                model="mock:test",
                ctx=2048,
                ngpu=0,
                max_steps=3,
                max_wall=0.5,
                run_dir=os.path.join(tmp, "tool_timeout_run"),
                actuator="native_tools",
                verbose=False,
                chat_fn=custom_chat_fn,
            )
            created_file = os.path.join(tmp, "tool_timeout_run", "work", "should_never_exist_timeout.txt")
            if os.path.exists(created_file):
                failures.append(("inference_deadline", "incomplete tool call was executed!", ""))
            if run_res.get("outcome") != "timeout":
                failures.append(("inference_deadline", f"expected outcome 'timeout', got {run_res.get('outcome')}", str(run_res)))
            if not run_res.get("symptoms", {}).get("timeout_exceeded"):
                failures.append(("inference_deadline", "symptom timeout_exceeded not True", str(run_res)))
            if not run_res.get("tokens_incomplete"):
                failures.append(("inference_deadline", "run_res tokens_incomplete not True", str(run_res)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # 5. Normal requests finishing before the deadline
        server.mode = "normal"
        t0 = time.monotonic()
        res5 = chat_stream("mock", [{"role": "user", "content": "hi"}], 2048, 0, timeout=2.0, url=url)
        dur5 = time.monotonic() - t0
        if res5.telemetry.get("incomplete_stream"):
            failures.append(("inference_deadline", "normal request: incomplete_stream was True", str(res5.telemetry)))
        if res5.telemetry.get("tokens_incomplete"):
            failures.append(("inference_deadline", "normal request: tokens_incomplete was True", str(res5.telemetry)))
        if res5.telemetry.get("done_reason") != "stop":
            failures.append(("inference_deadline", f"normal request: done_reason != stop ({res5.telemetry.get('done_reason')})", str(res5.telemetry)))
        if res5.content != "Hello world":
            failures.append(("inference_deadline", f"normal request: content mismatch ({res5.content})", ""))

        # 6. Cleanup of cancelled request and absence of lingering inference activity
        active_children = mp.active_children()
        try:
            fs_children = mp.get_context("forkserver").active_children()
        except ValueError:
            fs_children = []
        if active_children or fs_children:
            failures.append(("inference_deadline", f"lingering worker child processes found: mp={active_children} fs={fs_children}", ""))

        # 7. Malformed chunk handling and cleanup
        server.mode = "malformed_chunk"
        try:
            chat_stream("mock", [{"role": "user", "content": "hi"}], 2048, 0, timeout=1.0, url=url)
            failures.append(("inference_deadline", "malformed chunk: did not raise RuntimeError", ""))
        except RuntimeError as e:
            if "Malformed stream chunk" not in str(e):
                failures.append(("inference_deadline", f"malformed chunk: unexpected error message: {e}", ""))
    finally:
        server.shutdown()
        server.server_close()

    return failures


def classifier_unit_tests() -> list[tuple[str, str, str]]:
    """Unit test failure classification outcome hierarchy and empirical cause rules."""
    failures = []

    # 1. Passed task has no failure cause but preserves diagnostic symptoms
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
    if "symptoms" not in c_pass or not isinstance(c_pass["symptoms"], dict):
        failures.append(("classifier", "symptoms missing on passed run", str(c_pass)))

    # 1b. Passed task with earlier invalid actions preserves invalid_actions symptom
    c_pass_invalid = classify_task_run(
        passed=True,
        timed_out=False,
        step_count=4,
        max_steps=20,
        invalid_actions=2,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="OK",
        steps=[],
    )
    if c_pass_invalid["outcome"] != "passed":
        failures.append(
            (
                "classifier",
                "passed with invalid actions outcome error",
                str(c_pass_invalid),
            )
        )
    if not c_pass_invalid.get("symptoms", {}).get("invalid_actions_detected"):
        failures.append(
            (
                "classifier",
                "invalid_actions_detected symptom not set on passed run",
                str(c_pass_invalid),
            )
        )

    # 1c. Inference error overrides passed outcome -> inference_failure
    c_pass_err = classify_task_run(
        passed=True,
        timed_out=False,
        step_count=2,
        max_steps=20,
        invalid_actions=0,
        forbidden_files=[],
        changed_files=["app.py"],
        exit_status="OK",
        steps=[],
        inference_error="transport stream closed",
    )
    if c_pass_err["outcome"] != "inference_failure":
        failures.append(
            (
                "classifier",
                "inference_error did not override passed outcome",
                str(c_pass_err),
            )
        )

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

    return failures


def classifier_empirical_cause_tests() -> list[tuple[str, str, str]]:
    """Unit test empirical cause attribution and symptom extraction rules."""
    failures = []

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

    # 6. Context exhaustion -> symptom context_capacity_reached=True, cause unknown
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
    if not c_ctx["symptoms"].get("context_capacity_reached"):
        failures.append(
            ("classifier", "context_capacity_reached symptom missing", str(c_ctx))
        )
    if c_ctx["possible_cause"] != "unknown":
        failures.append(
            ("classifier", "context exhaustion should not guess cause", str(c_ctx))
        )

    # 7. Length limit hit -> symptom output_limit_reached=True, cause unknown
    c_len = classify_task_run(
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
        telemetry_records=[{"prompt_eval_count": 1200, "done_reason": "length"}],
    )
    if not c_len["symptoms"].get("output_limit_reached"):
        failures.append(
            ("classifier", "output_limit_reached symptom missing", str(c_len))
        )
    if c_len["possible_cause"] != "unknown":
        failures.append(
            ("classifier", "output limit should not guess cause", str(c_len))
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


def gitignore_unit_tests() -> list[tuple[str, str, str]]:
    """Ensure benchmark runs and artifacts are strictly ignored by git."""
    failures = []
    test_paths = [
        "runs/test-run/transcript.json",
        "bench/runs/test-run/results.json",
        "nested/sub/runs/run-1/env.json",
    ]
    for path in test_paths:
        res = subprocess.run(
            ["git", "-C", REPO_ROOT, "check-ignore", path],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            failures.append(("gitignore", f"path {path} is not gitignored", res.stderr))
    return failures


def sandbox_python_assertion_tests() -> list[tuple[str, str, str]]:
    """Ensure sandbox Python interpreter runs and matches host version."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-pyassert-")
    try:
        py_info = assert_sandbox_python(tmp)
        host_ver = sys.version.split()[0]
        sb_ver = py_info["sandbox_python_version"]
        if sb_ver != host_ver:
            failures.append(
                (
                    "sandbox_python",
                    f"exact version mismatch: sandbox={sb_ver} host={host_ver}",
                    "",
                )
            )
        if not py_info.get("sandbox_python_executable"):
            failures.append(
                (
                    "sandbox_python",
                    "sandbox_python_executable path missing",
                    str(py_info),
                )
            )
    except Exception as e:
        failures.append(("sandbox_python", f"assertion raised error: {e}", ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def native_tool_fairness_unit_tests() -> list[tuple[str, str, str]]:
    """Test strict native tools actuator: budget parity, finish batch rule, timeout."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-tool-fairness-")
    meta = load_task("debugging-default-arg")
    try:
        call_history: list[list[dict[str, Any]]] = []

        def mock_multi_call(*args: Any, **kwargs: Any) -> StreamResult:
            msgs = args[1]
            call_history.append(list(msgs))
            if len(msgs) <= 2:
                return StreamResult(
                    content="",
                    thinking="planning commands...",
                    tool_calls=[
                        {
                            "id": "c1",
                            "function": {
                                "name": "execute_command",
                                "arguments": {"command": "echo 'first'"},
                            },
                        },
                        {
                            "id": "c2",
                            "function": {
                                "name": "execute_command",
                                "arguments": {"command": "echo 'second'"},
                            },
                        },
                    ],
                    telemetry={"eval_count": 10, "prompt_eval_count": 20},
                )
            return StreamResult(
                content="",
                thinking="",
                tool_calls=[
                    {"id": "c3", "function": {"name": "finish_task", "arguments": {}}}
                ],
                telemetry={"eval_count": 5, "prompt_eval_count": 10},
            )

        run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=5,
            max_wall=60,
            run_dir=os.path.join(tmp, "multi_call"),
            actuator="native_tools",
            verbose=False,
            chat_fn=mock_multi_call,
        )

        if len(call_history) < 2:
            failures.append(
                ("tool_fairness", "expected at least 2 chat turns for multi-call", "")
            )
        else:
            turn2_msgs = call_history[1]
            tool_results = [m for m in turn2_msgs if m.get("role") == "tool"]
            if len(tool_results) != 2:
                failures.append(
                    (
                        "tool_fairness",
                        f"expected 2 tool results in turn 2, got {len(tool_results)}",
                        str(tool_results),
                    )
                )
            tool_names = [m.get("tool_name") for m in tool_results]
            if tool_names != ["execute_command", "execute_command"]:
                failures.append(
                    ("tool_fairness", f"tool_names mismatch: {tool_names}", "")
                )
            tool_ids = [m.get("tool_call_id") for m in tool_results]
            if tool_ids != ["c1", "c2"]:
                failures.append(
                    ("tool_fairness", f"tool_call_ids mismatch: {tool_ids}", "")
                )

        # 2. Command execution budget parity test
        # Model emits 3 commands in 1 turn, but max_steps (and thus max_commands) is 2
        def mock_budget_overflow(*args: Any, **kwargs: Any) -> StreamResult:
            return StreamResult(
                content="",
                thinking="",
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": "echo 'cmd1'"},
                        },
                    },
                    {
                        "id": "c2",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": "echo 'cmd2'"},
                        },
                    },
                    {
                        "id": "c3",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": "touch cmd3_should_not_run.txt"},
                        },
                    },
                ],
                telemetry={"eval_count": 10, "prompt_eval_count": 10},
            )

        rec_budget = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=2,
            max_wall=60,
            run_dir=os.path.join(tmp, "budget_test"),
            actuator="native_tools",
            verbose=False,
            chat_fn=mock_budget_overflow,
        )
        if rec_budget["commands_executed"] != 2:
            failures.append(
                (
                    "tool_fairness",
                    f"expected 2 commands, got {rec_budget['commands_executed']}",
                    str(rec_budget),
                )
            )
        if os.path.exists(
            os.path.join(tmp, "budget_test", "work", "cmd3_should_not_run.txt")
        ):
            failures.append(
                (
                    "tool_fairness",
                    "command exceeding max_commands budget was executed!",
                    "",
                )
            )

        # 3. Reject mixed finish_task batch
        def mock_mixed_finish(*args: Any, **kwargs: Any) -> StreamResult:
            return StreamResult(
                content="",
                thinking="",
                tool_calls=[
                    {
                        "id": "c_cmd",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": "touch mixed_should_not_run.txt"},
                        },
                    },
                    {
                        "id": "c_fin",
                        "function": {"name": "finish_task", "arguments": {}},
                    },
                ],
                telemetry={"eval_count": 10, "prompt_eval_count": 10},
            )

        rec_mixed = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=2,
            max_wall=60,
            run_dir=os.path.join(tmp, "mixed_finish"),
            actuator="native_tools",
            verbose=False,
            chat_fn=mock_mixed_finish,
        )
        if rec_mixed["invalid_actions"] == 0:
            failures.append(
                (
                    "tool_fairness",
                    "batch mixing finish_task and command was not rejected",
                    str(rec_mixed),
                )
            )
        if os.path.exists(
            os.path.join(tmp, "mixed_finish", "work", "mixed_should_not_run.txt")
        ):
            failures.append(
                (
                    "tool_fairness",
                    "command from mixed finish_task batch was executed!",
                    "",
                )
            )

        # 4. Plain text in native tools mode rejection
        def mock_text_in_tools(*args: Any, **kwargs: Any) -> StreamResult:
            return StreamResult(
                content="RUN: touch should_not_exist.txt",
                thinking="",
                tool_calls=[],
                telemetry={"eval_count": 10, "prompt_eval_count": 10},
            )

        rec_reject = run_task(
            meta=meta,
            model="mock:test",
            ctx=2048,
            ngpu=0,
            max_steps=2,
            max_wall=60,
            run_dir=os.path.join(tmp, "reject_text"),
            actuator="native_tools",
            verbose=False,
            chat_fn=mock_text_in_tools,
        )
        if rec_reject["invalid_actions"] == 0:
            failures.append(
                (
                    "tool_fairness",
                    "plain text in native_tools was not flagged as invalid",
                    str(rec_reject),
                )
            )
        if os.path.exists(
            os.path.join(tmp, "reject_text", "work", "should_not_exist.txt")
        ):
            failures.append(
                (
                    "tool_fairness",
                    "plain text RUN: command was executed in native_tools mode!",
                    "",
                )
            )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def policy_check_symlink_tests() -> list[tuple[str, str, str]]:
    """Verify policy_check rejects symlinks, even at allowlisted paths."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="bench-symlink-policy-")
    try:
        run_dir = os.path.join(tmp, "run")
        work_dir = os.path.join(run_dir, "work")
        os.makedirs(work_dir)

        # Initial baseline snapshot with allowlisted file
        baseline_file = os.path.join(work_dir, "app.py")
        with open(baseline_file, "w") as f:
            f.write("# original app\n")

        with open(os.path.join(run_dir, "baseline.json"), "w") as f:
            json.dump(snapshot(work_dir), f)

        # Replace allowlisted file with a symlink
        target_file = os.path.join(tmp, "secret.txt")
        with open(target_file, "w") as f:
            f.write("SECRET")

        os.remove(baseline_file)
        os.symlink(target_file, baseline_file)

        meta = {"allowed_files": ["app.py"]}
        diff = policy_check(meta, work_dir, run_dir)
        if "app.py" not in diff["forbidden"]:
            failures.append(
                (
                    "policy_check",
                    "symlink at allowlisted path 'app.py' was not flagged as forbidden",
                    str(diff),
                )
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


def cgroup_background_child_termination_tests() -> list[tuple[str, str, str]]:
    """Test background subprocess termination and cgroup cleanup failure reporting."""
    failures: list[tuple[str, str, str]] = []
    tmp = tempfile.mkdtemp(prefix="bench-cgroup-child-")
    try:
        work = os.path.join(tmp, "work")
        os.makedirs(work, exist_ok=True)

        # 1. Background child must not survive command timeout
        bg_sleep_dur = 47
        cmd = (
            f'python3 -c "import subprocess, time; '
            f"subprocess.Popen(['sleep', '{bg_sleep_dur}']); time.sleep(5)\""
        )
        code, out = run_sandboxed(work, cmd, timeout=1)
        if code != -1:
            failures.append(
                ("cgroup_cleanup", f"expected timeout exit -1, got {code}", out)
            )
        if "[bench: command timed out after 1s]" not in out:
            failures.append(
                ("cgroup_cleanup", "missing timeout message in output", out)
            )
        if "[bench: cgroup cleanup failure:" in out:
            failures.append(
                ("cgroup_cleanup", f"unexpected cleanup failure reported: {out}", "")
            )

        time.sleep(0.3)
        check_proc = subprocess.run(
            ["pgrep", "-f", f"sleep {bg_sleep_dur}"],
            capture_output=True,
            text=True,
        )
        pids = check_proc.stdout.strip()
        if check_proc.returncode == 0 and pids:
            failures.append(
                (
                    "cgroup_cleanup",
                    f"background child process survived timeout: PIDs {pids}",
                    "",
                )
            )

        # 2. Verify cleanup failure reporting when _kill_cgroup_scope fails
        def _mock_fail(unit_name: str, timeout: float = 5.0) -> tuple[bool, str]:
            return False, "mocked scope kill failure"

        attr = "_kill_cgroup_scope"
        orig_kill = bench_mod._kill_cgroup_scope
        try:
            setattr(bench_mod, attr, _mock_fail)
            _, out_fail = run_sandboxed(work, "sleep 5", timeout=1)
            if (
                "[bench: cgroup cleanup failure: mocked scope kill failure]"
                not in out_fail
            ):
                failures.append(
                    (
                        "cgroup_cleanup",
                        "failed to report cleanup failure in output",
                        out_fail,
                    )
                )
        finally:
            setattr(bench_mod, attr, orig_kill)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return failures


def sandbox_output_symlink_regression_test() -> list[tuple[str, str, str]]:
    """Verify untrusted sandbox cannot disclose host files via output symlinks."""
    failures: list[tuple[str, str, str]] = []
    tmp = tempfile.mkdtemp(prefix="bench-sec-sentinel-")
    try:
        sentinel_dir = os.path.join(tmp, "host_private")
        os.makedirs(sentinel_dir, exist_ok=True)
        sentinel_file = os.path.join(sentinel_dir, "secret.key")
        sentinel_secret = "HARMLESS_HOST_SENTINEL_TOKEN_SECRET_987654"
        with open(sentinel_file, "w") as f:
            f.write(sentinel_secret)

        work = os.path.join(tmp, "work")
        os.makedirs(work, exist_ok=True)
        cmd_out_dir = os.path.join(tmp, "cmd_out")
        os.makedirs(cmd_out_dir, exist_ok=True)

        attack_cmd = (
            f"mkdir -p /mnt/out 2>/dev/null || true; "
            f"ln -sf '{sentinel_file}' /mnt/out/stdout 2>/dev/null || true; "
            f"ln -sf '{sentinel_file}' /mnt/work/stdout 2>/dev/null || true; "
            f"echo 'normal agent output'"
        )

        _, out = run_sandboxed(work, attack_cmd, timeout=10, outdir=cmd_out_dir)

        if sentinel_secret in out:
            failures.append(
                (
                    "sandbox_output_security",
                    "host sentinel content was disclosed through sandboxed output",
                    out,
                )
            )
        if "normal agent output" not in out:
            failures.append(
                (
                    "sandbox_output_security",
                    "normal sandboxed output was not captured",
                    out,
                )
            )

        mount_probe_cmd = (
            "touch /mnt/out/test_leak.txt 2>/dev/null && "
            "echo 'OUTPUT_DIR_MOUNTED' || echo 'OUTPUT_DIR_UNMOUNTED'"
        )
        _, out_probe = run_sandboxed(
            work, mount_probe_cmd, timeout=10, outdir=cmd_out_dir
        )
        if "OUTPUT_DIR_MOUNTED" in out_probe or os.path.exists(
            os.path.join(cmd_out_dir, "test_leak.txt")
        ):
            failures.append(
                (
                    "sandbox_output_security",
                    "sandbox was able to write directly into host output directory",
                    out_probe,
                )
            )

        overwrite_cmd = (
            "echo 'echo PWNED' > /bench-run.sh 2>/dev/null && "
            "echo 'WRITABLE' || echo 'READONLY'"
        )
        _, out_ro = run_sandboxed(work, overwrite_cmd, timeout=10, outdir=cmd_out_dir)
        if "WRITABLE" in out_ro:
            failures.append(
                (
                    "sandbox_output_security",
                    "/bench-run.sh was writable from inside the sandbox",
                    out_ro,
                )
            )
    finally:
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

    print("Running premature stream closure tests...")
    failures.extend(premature_stream_closure_unit_tests())

    print("Running retry stream & scoring integrity tests...")
    failures.extend(retry_stream_and_scoring_integrity_tests())

    print("Running inference deadline regression tests...")
    failures.extend(inference_deadline_regression_tests())

    print("Running classifier unit tests...")
    failures.extend(classifier_unit_tests())
    failures.extend(classifier_empirical_cause_tests())

    print("Running strict isolation checks...")
    failures.extend(strict_isolation_test())

    print("Running isolation attack checks...")
    failures.extend(isolation_checks())

    print("Running subprocess resume integration tests...")
    failures.extend(subprocess_resume_integration_test())

    print("Running actuator smoke tests (text and native_tools)...")
    failures.extend(runner_actuator_smoke_tests())

    print("Running gitignore unit tests...")
    failures.extend(gitignore_unit_tests())

    print("Running sandbox python assertion tests...")
    failures.extend(sandbox_python_assertion_tests())

    print("Running native tool fairness tests...")
    failures.extend(native_tool_fairness_unit_tests())

    print("Running symlink policy check tests...")
    failures.extend(policy_check_symlink_tests())

    print("Running sandbox regression checks...")
    failures.extend(sandbox_regression_checks())

    print("Running cgroup background child termination tests...")
    failures.extend(cgroup_background_child_termination_tests())

    print("Running sandbox output symlink security tests...")
    failures.extend(sandbox_output_symlink_regression_test())

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
