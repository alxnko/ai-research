"""Self-tests for the safeguards themselves.

A safeguard that passes regardless of input gives no assurance. Each scenario
here feeds a safeguard one injected defect and its clean twin: the defect must
be detected and the clean input must pass. Everything runs in disposable
directories: no GPU, no Ollama, no network beyond a loopback listener the test
owns, and no write to runs/ or any frozen artifact.

Entry point: validator_selftests(check_task), called from bench/selfcheck.py.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any
from unittest import mock

from bench import bench as bench_mod
from bench.bench import install_tests, load_task, policy_check, run_sandboxed
from experiments import analyze_exp2 as analyzer
from experiments import exp2_entry as entry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Failure = tuple[str, str, str]


def _expect(out: list[Failure], scenario: str, ok: bool, why: str, detail: str = ""):
    if not ok:
        out.append((scenario, why, detail[-600:]))


def _pinned_version(tool: str) -> str | None:
    """The version check.sh actually runs, so the test cannot drift from it."""
    with open(os.path.join(REPO_ROOT, "check.sh"), encoding="utf-8") as f:
        match = re.search(rf"uvx {tool}@([0-9][0-9.]*)", f.read())
    return match.group(1) if match else None


def _uvx(tool: str, args: list[str], cwd: str) -> tuple[int, str]:
    version = _pinned_version(tool)
    if version is None:
        return -1, f"check.sh no longer pins {tool}; update validator_selftest"
    proc = subprocess.run(
        ["uvx", f"{tool}@{version}", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# 1 + 2. Static gates ---------------------------------------------------------


def ruff_and_ty_gates() -> list[Failure]:
    out: list[Failure] = []
    tmp = tempfile.mkdtemp(prefix="selftest-static-")
    try:
        shutil.copy(os.path.join(REPO_ROOT, "pyproject.toml"), tmp)  # real rules
        _write(os.path.join(tmp, "clean.py"), "x = 1\n")
        _write(os.path.join(tmp, "unused.py"), "import os\n")  # F401
        _write(os.path.join(tmp, "unformatted.py"), "x=1\n")

        rc, text = _uvx("ruff", ["check", "--no-cache", "clean.py"], tmp)
        _expect(out, "ruff", rc == 0, "clean file was rejected by lint", text)
        rc, text = _uvx("ruff", ["check", "--no-cache", "unused.py"], tmp)
        _expect(out, "ruff", rc != 0 and "F401" in text, "unused import passed", text)
        rc, text = _uvx("ruff", ["format", "--check", "clean.py"], tmp)
        _expect(out, "ruff", rc == 0, "formatted file was rejected", text)
        rc, text = _uvx("ruff", ["format", "--check", "unformatted.py"], tmp)
        _expect(out, "ruff", rc != 0, "unformatted file passed the format check", text)

        # ty gets its own minimal project: the repo's extra-paths do not exist here
        os.remove(os.path.join(tmp, "pyproject.toml"))
        _write(
            os.path.join(tmp, "pyproject.toml"),
            '[project]\nname = "t"\nversion = "0"\n',
        )
        _write(os.path.join(tmp, "typed.py"), "x: int = 1\n")
        _write(os.path.join(tmp, "wrong.py"), 'x: int = "a"\n')
        rc, text = _uvx("ty", ["check", "typed.py"], tmp)
        _expect(out, "ty", rc == 0, "well-typed file was rejected", text)
        rc, text = _uvx("ty", ["check", "wrong.py"], tmp)
        _expect(out, "ty", rc != 0, "type error passed", text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# 3. Scorer / fixture tampering -----------------------------------------------


def _tampered_workspace(tmp: str) -> tuple[dict[str, Any], str, str]:
    run_dir = os.path.join(tmp, "run")
    work = os.path.join(run_dir, "work")
    os.makedirs(work)
    _write(os.path.join(work, "app.py"), "# app\n")
    _write(os.path.join(work, "test_public.py"), "# public test\n")
    meta = {"allowed_files": ["app.py"]}
    with open(os.path.join(run_dir, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump(bench_mod.snapshot(work), f)
    return meta, run_dir, work


def scorer_tamper_gates(check_task: Callable) -> list[Failure]:
    out: list[Failure] = []
    tmp = tempfile.mkdtemp(prefix="selftest-tamper-")
    try:
        meta, run_dir, work = _tampered_workspace(tmp)

        _write(os.path.join(work, "app.py"), "# edited\n")  # clean: allowed edit
        diff = policy_check(meta, work, run_dir)
        _expect(
            out, "tamper", diff["forbidden"] == [], "allowed edit flagged", str(diff)
        )

        for label, mutate in (
            ("edited test", lambda: _write(os.path.join(work, "test_public.py"), "x")),
            ("added file", lambda: _write(os.path.join(work, "conftest.py"), "x")),
            ("removed test", lambda: os.remove(os.path.join(work, "test_public.py"))),
        ):
            mutate()
            diff = policy_check(meta, work, run_dir)
            _expect(out, "tamper", bool(diff["forbidden"]), f"{label} not flagged")

        # A workspace that pre-plants _hidden must not influence scoring: the
        # scorer rebuilds from regular files outside _hidden, then installs
        # the real tests. A naive copy is the control that shows the leak.
        task = load_task("debugging-off-by-one")
        planted = os.path.join(tmp, "planted")
        os.makedirs(os.path.join(planted, "_hidden"))
        _write(os.path.join(planted, "_hidden", "test_planted.py"), "# planted\n")
        naive = os.path.join(tmp, "naive-copy")
        shutil.copytree(planted, naive)
        _expect(
            out,
            "tamper",
            os.path.exists(os.path.join(naive, "_hidden", "test_planted.py")),
            "control: naive copy should carry the planted file",
        )
        scoring = os.path.join(tmp, "scoring")
        bench_mod.clean_copy(planted, scoring)
        try:
            installed = install_tests(task, scoring)
        except RuntimeError as exc:
            installed = False
            out.append(("tamper", "planted _hidden reached scoring", str(exc)))
        if installed:
            _expect(
                out,
                "tamper",
                not os.path.exists(os.path.join(scoring, "_hidden", "test_planted.py")),
                "planted _hidden content reached the scoring workspace",
            )

        out.extend(_fixture_tamper(tmp, check_task))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _task_copy(tmp: str, name: str) -> dict[str, Any]:
    meta = load_task("debugging-off-by-one")
    dst = os.path.join(tmp, name)
    shutil.copytree(meta["dir"], dst)
    meta.update(
        dir=dst,
        task_dir=dst,
        tests_dir=os.path.join(dst, "tests"),
        repo_dir=os.path.join(dst, "repo"),
        solution_dir=os.path.join(dst, "solution"),
    )
    return meta


def _fixture_tamper(tmp: str, check_task: Callable) -> list[Failure]:
    """The golden check must flag a broken solution and a vacuous test suite."""
    out: list[Failure] = []
    clean = _task_copy(tmp, "fx-clean")
    broken = _task_copy(tmp, "fx-broken-solution")
    for name in os.listdir(broken["solution_dir"]):  # solution reverts to the bug
        shutil.copy2(
            os.path.join(broken["repo_dir"], name),
            os.path.join(broken["solution_dir"], name),
        )
    vacuous = _task_copy(tmp, "fx-vacuous-tests")
    for dirpath, _dirs, files in os.walk(vacuous["tests_dir"]):
        for name in files:
            if name.startswith("test_") and name.endswith(".py"):
                _write(
                    os.path.join(dirpath, name),
                    "import unittest\n\n\nclass T(unittest.TestCase):\n"
                    "    def test_ok(self):\n        self.assertTrue(True)\n",
                )
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            key: pool.submit(check_task, "debugging-off-by-one", meta)
            for key, meta in (
                ("clean", clean),
                ("broken", broken),
                ("vacuous", vacuous),
            )
        }
        results = {key: fut.result()[1] for key, fut in futures.items()}
    _expect(
        out,
        "fixture_tamper",
        not results["clean"],
        "clean copy flagged",
        str(results["clean"]),
    )
    _expect(
        out,
        "fixture_tamper",
        any("solution does not pass" in why for _t, why, _o in results["broken"]),
        "golden check missed a broken reference solution",
    )
    _expect(
        out,
        "fixture_tamper",
        any("no work" in why for _t, why, _o in results["vacuous"]),
        "golden check missed vacuous tests",
    )
    return out


# 4. Model fingerprint ---------------------------------------------------------


def fingerprint_gate() -> list[Failure]:
    out: list[Failure] = []
    tmp = tempfile.mkdtemp(prefix="selftest-fingerprint-")
    bench = entry.benchmark
    run_dir = os.path.join(
        tmp, "runs", "exp2", "control", "seed42", "backend-auth-token"
    )
    argv = [
        "exp2_entry", "--exp2-policy", "control", "--run-dir", run_dir,
        "--model", "qwen3.5:4b-q4_K_M", "--actuator", "native_tools",
        "--task", "backend-auth-token", "--ctx", "16384", "--ngpu", "99",
        "--steps", "20", "--max-wall", "600", "--strict-isolation",
        "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
        "--repeat-penalty", "1.0", "--seed", "42",
        "--sampling-profile", "exp2-coding-v1",
    ]  # fmt: skip
    good = {**entry.EXPECTED_MODEL, "config_digest": "sha256:cfg", "weights_bytes": 1}

    def run(fingerprint: dict[str, Any]) -> tuple[str, str]:
        patches = (
            mock.patch.object(bench, "ROOT", tmp),
            mock.patch.object(bench, "main", return_value=0),
            mock.patch.object(bench, "model_fingerprint", return_value=fingerprint),
            mock.patch.object(sys, "argv", argv[:]),
            contextlib.redirect_stdout(io.StringIO()),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            try:
                return f"exit={entry.main()}", ""
            except SystemExit as exc:
                return "refused", str(exc)

    try:
        wrong = {**good, "weights_digest": "sha256:" + "0" * 64}
        status, msg = run(wrong)
        _expect(
            out,
            "fingerprint",
            status == "refused" and "identity mismatch" in msg,
            "wrong weights digest was not refused",
            f"{status} {msg}",
        )
        _expect(
            out,
            "fingerprint",
            not os.path.exists(run_dir),
            "refused run still created its output directory",
        )
        status, msg = run(good)
        _expect(out, "fingerprint", status == "exit=0", "registered model refused", msg)
        # resume with any other fingerprint field must be refused too
        status, msg = run({**good, "config_digest": "sha256:other"})
        _expect(
            out,
            "fingerprint",
            status == "refused" and "different policy" in msg,
            "resume with a drifted fingerprint was accepted",
            f"{status} {msg}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# 5. Missing experiment result --------------------------------------------------


def _fake_results(root: str, ok: Callable[[str, int, str], bool]) -> None:
    for arm in analyzer.ARMS:
        for seed in analyzer.SEEDS:
            for task in analyzer.TASKS:
                d = os.path.join(root, arm, f"seed{seed}", task)
                os.makedirs(d)
                row = {
                    "task": task,
                    "passed": ok(arm, seed, task),
                    "run_successful": ok(arm, seed, task),
                    "total_wall_s": 30.0,
                    "tokens": 1000,
                }
                _write(os.path.join(d, "results.json"), json.dumps([row]))


def _analyze(root: str, *flags: str) -> tuple[int, str]:
    buf = io.StringIO()
    with (
        mock.patch.object(analyzer, "RUNS", root),
        mock.patch.object(sys, "argv", ["analyze_exp2", *flags]),
        contextlib.redirect_stdout(buf),
    ):
        rc = analyzer.main()
    return rc, buf.getvalue()


def result_completeness_gates() -> list[Failure]:
    """Analyzer contract: it REPORTS gaps and withholds the decision (exit 0).
    Enforcement is run_exp2.sh validate_result, tested separately below."""
    out: list[Failure] = []
    tmp = tempfile.mkdtemp(prefix="selftest-results-")
    try:
        root = os.path.join(tmp, "exp2")
        _fake_results(
            root,
            lambda arm, _s, task: (
                arm == "verification" or task == "debugging-off-by-one"
            ),
        )
        rc, text = _analyze(root)
        _expect(
            out,
            "results",
            rc == 0
            and "attempts found: 36/36" in text
            and "registered decision rule met" in text,
            "complete data did not yield a decision",
            text,
        )

        victim = os.path.join(root, "control", "seed43", "refactor-parser")
        os.remove(os.path.join(victim, "results.json"))
        rc, text = _analyze(root)
        _expect(
            out,
            "results",
            "attempts found: 35/36" in text
            and "INCOMPLETE" in text
            and "registered decision rule met" not in text,
            "missing result still produced a decision",
            text,
        )
        _expect(out, "results", rc == 0, "analyzer contract changed: it is report-only")
        _, js = _analyze(root, "--json")
        _expect(
            out,
            "results",
            ["control", 43, "refactor-parser"] in json.loads(js)["missing"],
            "--json did not list the missing cell",
            js,
        )

        _write(os.path.join(victim, "results.json"), json.dumps([{"task": "x"}]))
        _, text = _analyze(root)
        _expect(
            out,
            "results",
            "35/36" in text and "INCOMPLETE" in text,
            "malformed record counted as a valid result",
            text,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.extend(_runner_validation_gate())
    return out


def _runner_validation_gate() -> list[Failure]:
    """run_exp2.sh validate_result is what actually halts a sweep."""
    out: list[Failure] = []
    with open(
        os.path.join(REPO_ROOT, "experiments", "run_exp2.sh"), encoding="utf-8"
    ) as f:
        match = re.search(r"^validate_result\(\) \{.*?^\}", f.read(), re.S | re.M)
    if not match:
        return [("runner", "validate_result() not found in run_exp2.sh", "")]
    tmp = tempfile.mkdtemp(prefix="selftest-runner-")
    row = {
        "task": "t",
        "passed": 1,
        "run_successful": 1,
        "total_wall_s": 1,
        "tokens": 1,
    }
    cases = {
        "valid": (json.dumps([row]), 0),
        "missing keys": (json.dumps([{"task": "t"}]), 1),
        "two records": (json.dumps([row, row]), 1),
        "wrong task": (json.dumps([{**row, "task": "other"}]), 1),
        "corrupt json": ("{not json", 1),
        "absent file": (None, 1),
    }
    try:
        for label, (body, want) in cases.items():
            d = os.path.join(tmp, label.replace(" ", "_"))
            os.makedirs(d)
            if body is not None:
                _write(os.path.join(d, "results.json"), body)
            proc = subprocess.run(
                ["bash", "-c", f'{match.group(0)}\nvalidate_result "$1" t', "_", d],
                capture_output=True,
                text=True,
                timeout=30,
            )
            _expect(
                out,
                "runner",
                proc.returncode == want,
                f"validate_result({label}) exit {proc.returncode}, want {want}",
                proc.stdout + proc.stderr,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# 6. Sandbox escape --------------------------------------------------------------


def _loopback_listener() -> tuple[list[int], int, Callable[[], None]]:
    """A loopback TCP listener the test owns; `hits` records each connection."""
    hits: list[int] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.2)
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            hits.append(1)
            conn.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()

    def shutdown() -> None:
        stop.set()
        listener.close()
        thread.join(timeout=2)

    return hits, listener.getsockname()[1], shutdown


def sandbox_escape_gates() -> list[Failure]:
    """Probe host canary read, write outside the workspace, and network egress.
    The same probe run unsandboxed must light every detector, so a probe that
    cannot see a leak cannot pass for isolation."""
    out: list[Failure] = []
    tmp = tempfile.mkdtemp(prefix="selftest-escape-")
    token = "HOST_CANARY_" + os.urandom(6).hex()
    canary = os.path.join(tmp, "host_canary.txt")
    _write(canary, token)
    work = os.path.join(tmp, "work")
    os.makedirs(work)

    hits, port, stop_listener = _loopback_listener()

    probe = (
        "echo PROBE_RAN; "
        f"cat '{canary}' 2>/dev/null; "
        f"echo pwned > '{tmp}/escaped.txt' 2>/dev/null; "
        f"(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null; true"
    )

    def leaks(text: str) -> dict[str, bool]:
        time.sleep(0.4)  # let the listener thread record a connection
        return {
            "read host canary": token in text,
            "wrote outside workspace": os.path.exists(f"{tmp}/escaped.txt"),
            "reached host loopback": bool(hits),
        }

    try:
        # control: unsandboxed, every detector must fire
        proc = subprocess.run(
            ["bash", "-c", probe], capture_output=True, text=True, timeout=20
        )
        control = leaks(proc.stdout)
        skipped = set()
        for name, fired in control.items():
            if not fired and name == "reached host loopback":
                # host loopback is down (e.g. a network-less namespace): the
                # probe cannot be shown sensitive, so say so instead of passing
                skipped.add(name)
                print(
                    f"  SKIPPED sandbox network probe: {name} (control cannot connect)"
                )
                continue
            _expect(
                out,
                "sandbox_escape",
                fired,
                f"probe cannot detect: {name}",
                proc.stdout,
            )
        if os.path.exists(f"{tmp}/escaped.txt"):
            os.remove(f"{tmp}/escaped.txt")
        hits.clear()

        # sandboxed: same probe, nothing may leak and the probe must have run
        _, text = run_sandboxed(work, probe, timeout=20)
        _expect(
            out,
            "sandbox_escape",
            "PROBE_RAN" in text,
            "probe did not run in sandbox",
            text,
        )
        for name, leaked in leaks(text).items():
            if name in skipped:
                continue
            _expect(out, "sandbox_escape", not leaked, f"sandbox escape: {name}", text)
    finally:
        stop_listener()
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def validator_selftests(check_task: Callable) -> list[Failure]:
    out: list[Failure] = []
    steps: list[tuple[str, Callable[[], list[Failure]]]] = [
        ("ruff/ty gates", ruff_and_ty_gates),
        ("scorer/fixture tamper", lambda: scorer_tamper_gates(check_task)),
        ("model fingerprint", fingerprint_gate),
        ("missing results", result_completeness_gates),
        ("sandbox escape", sandbox_escape_gates),
    ]
    for label, fn in steps:
        t0 = time.monotonic()
        found = fn()
        status = "FAIL" if found else "ok"
        print(f"  {status:5s} {label} ({time.monotonic() - t0:.1f}s)")
        out.extend(found)
    return out
