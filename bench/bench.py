"""Agentic coding benchmark runner for local LLMs in a secure Linux sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.classifier import classify_task_run
from bench.telemetry import (
    BENCH_TOOLS,
    SamplingConfig,
    StreamResult,
    VramMonitor,
    chat_stream,
    check_gpu_telemetry,
    get_ollama_offload_status,
    summarize_inference,
)

TASKS = os.path.join(ROOT, "bench", "tasks")
RUNS = os.path.join(ROOT, "runs")

SYSTEM_PROMPT_TEXT = """
You are an agentic software engineer fixing a codebase in a sandbox.
At each turn, you can inspect or modify files by running bash commands.

To run a command, emit EXACTLY:
RUN: <command>

When the task is complete and verified:
DONE

Rules:
- Exactly one RUN or DONE per reply.
- Never edit the test runner or files you are not asked to modify.
- Do not run interactive commands (vi, nano, less, python without -c).
- Make sure all your changes are saved to disk before saying DONE.
- You have at most {max_steps} steps. Be concise.
"""

SYSTEM_PROMPT_TOOLS = """
You are an agentic software engineer fixing a codebase in a sandbox.
Use the execute_command tool to run bash commands in the workspace.
When all edits and testing are complete, call finish_task to submit the solution.

Rules:
- Never edit the test runner or files you are not asked to modify.
- Do not run interactive commands (vi, nano, less, python without -c).
- Ensure all changes are verified before finishing.
- You have at most {max_steps} steps. Be concise.
"""


def parse_action(reply: str | None) -> tuple[str | None, str | None]:
    """Extract agent action (run or done) from model text output."""
    if not reply:
        return None, None
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        if line == "DONE":
            return "done", None
    m = re.search(r"```(?:bash|sh)?\s*(.*?)\s*```", reply, re.DOTALL)
    if m:
        c = m.group(1).strip()
        if c:
            return "run", c
    m = re.search(
        r"<(?:command|result)>\s*(.*?)\s*</(?:command|result)>", reply, re.DOTALL
    )
    if m:
        c = m.group(1).strip()
        if c:
            return "run", c
    if "RUN:" in reply:
        idx = reply.index("RUN:") + 4
        cmd = reply[idx:].strip()
        if cmd.endswith("```"):
            cmd = cmd[:-3].strip()
        if cmd:
            return "run", cmd
    return None, None


ETC_FILES = [
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/localtime",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
]


def task_digest(task_id: str) -> str:
    """Compute sha256 fingerprint of task directory to ensure fixture integrity."""
    tdir = os.path.join(TASKS, task_id)
    if not os.path.isdir(tdir):
        return "unknown"
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(tdir)):
        for fn in sorted(files):
            fpath = os.path.join(root, fn)
            rel = os.path.relpath(fpath, tdir)
            h.update(rel.encode())
            try:
                with open(fpath, "rb") as f:
                    h.update(f.read())
            except OSError:
                pass
    return h.hexdigest()


def bwrap_cmd(
    workdir: str,
    inner_cmd: list[str],
    outdir: str | None = None,
    allow_network: bool = False,
) -> list[str]:
    """Assemble bubblewrap sandbox arguments."""
    pybase = os.environ.get("BENCH_PYTHON_HOME") or sys.base_prefix
    args = [
        "bwrap",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        pybase,
        "/opt/bench-python",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/root",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/etc",
        "--bind",
        workdir,
        "/mnt/work",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        "/mnt/work",
        "--unshare-pid",
        "--die-with-parent",
    ]
    if sys.prefix != sys.base_prefix:
        args.extend(["--ro-bind-try", sys.prefix, "/opt/bench-venv"])
    if not allow_network:
        args.append("--unshare-net")
    if outdir:
        args.extend(["--bind", outdir, "/mnt/out"])
    for ef in ETC_FILES:
        if os.path.exists(ef):
            args.extend(["--ro-bind", ef, ef])
    for ef_dir in ("/etc/ld.so.conf.d", "/usr/share/zoneinfo"):
        if os.path.isdir(ef_dir):
            args.extend(["--ro-bind", ef_dir, ef_dir])

    path_env = (
        "/opt/bench-venv/bin:/opt/bench-python/bin:/usr/bin:/bin"
        if sys.prefix != sys.base_prefix
        else "/opt/bench-python/bin:/usr/bin:/bin"
    )
    args.extend(
        [
            "--setenv",
            "HOME",
            "/root",
            "--setenv",
            "PATH",
            path_env,
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONUNBUFFERED",
            "1",
            "--setenv",
            "GIT_CONFIG_GLOBAL",
            "/dev/null",
            "--setenv",
            "GIT_CONFIG_SYSTEM",
            "/dev/null",
            "--setenv",
            "GIT_AUTHOR_NAME",
            "bench",
            "--setenv",
            "GIT_AUTHOR_EMAIL",
            "bench@local",
            "--setenv",
            "GIT_COMMITTER_NAME",
            "bench",
            "--setenv",
            "GIT_COMMITTER_EMAIL",
            "bench@local",
        ]
    )
    args.append("--")
    args.extend(inner_cmd)
    return args


def check_cgroup_v2() -> bool:
    """Check if unprivileged systemd user cgroup v2 limits are functional."""
    env = {
        "PATH": "/usr/bin:/bin:/opt/python/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if k in os.environ:
            env[k] = os.environ[k]
    try:
        res = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--scope",
                "-q",
                "-p",
                "MemoryMax=4G",
                "-p",
                "TasksMax=512",
                "true",
            ],
            capture_output=True,
            timeout=3,
            env=env,
        )
        return res.returncode == 0
    except OSError, subprocess.SubprocessError:
        return False


OUTPUT_CAP = 32768


def _read_output_tail(path: str, cap: int = OUTPUT_CAP) -> str:
    try:
        if not os.path.exists(path):
            return ""
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > cap:
                f.seek(size - cap)
                prefix = f"<output truncated, last {cap} bytes>\n"
            else:
                prefix = ""
            return prefix + f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _write_runner_script(
    script_path: str, stdout_path: str, cmd: str, use_cgroup: bool
) -> None:
    for path in (stdout_path, script_path):
        if os.path.exists(path):
            os.remove(path)
    normalized_cmd = cmd.rstrip("\r\n") + "\n"
    limits = [
        "ulimit -t 120 2>/dev/null || true",
        "ulimit -f 16384 2>/dev/null || true",
    ]
    if not use_cgroup:
        limits.append("ulimit -u 512 2>/dev/null || true")
    limits_str = "\n".join(limits) + "\n"
    script_content = (
        "#!/usr/bin/env bash\n"
        "exec > /mnt/out/stdout 2>&1\n"
        f"{limits_str}"
        f"{normalized_cmd}"
    )
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


def _build_sandbox_command(
    workdir: str,
    outdir: str,
    use_cgroup: bool,
    unit_name: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    base_bwrap = bwrap_cmd(workdir, ["bash", "/mnt/out/run.sh"], outdir)
    host_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if not use_cgroup:
        return base_bwrap, host_env

    for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if k in os.environ:
            host_env[k] = os.environ[k]
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        *(["--unit", unit_name] if unit_name else []),
        "-p",
        "MemoryMax=4G",
        "-p",
        "TasksMax=512",
        "--",
        *base_bwrap,
    ]
    return cmd, host_env


def run_sandboxed(
    workdir: str,
    cmd: str,
    timeout: int = 60,
    outdir: str | None = None,
    strict_isolation: bool = False,
) -> tuple[int, str]:
    """Execute command inside the bubblewrap sandbox with hard limits."""
    if outdir is None:
        outdir = os.path.join(workdir, ".bench_out")
    os.makedirs(outdir, exist_ok=True)
    stdout_path = os.path.join(outdir, "stdout")
    script_path = os.path.join(outdir, "run.sh")

    cgroup_ok = check_cgroup_v2()
    if strict_isolation and not cgroup_ok:
        raise RuntimeError(
            "Strict isolation failure: cgroup v2 systemd-run is required "
            "but not functional"
        )
    use_cgroup = cgroup_ok
    unit_name = f"bench-cmd-{uuid.uuid4().hex[:12]}" if use_cgroup else None
    _write_runner_script(script_path, stdout_path, cmd, use_cgroup)
    bwrap_full, host_env = _build_sandbox_command(
        workdir, outdir, use_cgroup, unit_name=unit_name
    )

    try:
        proc = subprocess.Popen(
            bwrap_full,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=host_env,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if use_cgroup and unit_name:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--signal=SIGKILL",
                        f"{unit_name}.scope",
                    ],
                    capture_output=True,
                )
            proc.kill()
            proc.wait()
            out = _read_output_tail(stdout_path)
            return -1, f"{out}\n[bench: command timed out after {timeout}s]"
        out = _read_output_tail(stdout_path)
        return proc.returncode, out
    except Exception as e:
        if strict_isolation and use_cgroup:
            raise RuntimeError(f"Strict isolation spawn failure: {e}") from e
        return -1, f"[bench: spawn error: {e}]"
    finally:
        if use_cgroup and unit_name:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--signal=SIGKILL",
                    f"{unit_name}.scope",
                ],
                capture_output=True,
            )


def assert_sandbox_python(
    workdir: str, strict_isolation: bool = False
) -> dict[str, str]:
    """Assert sandbox python runs, matches host version, and records executable."""
    code, out = run_sandboxed(
        workdir,
        'python3 -c "import sys; print(sys.version.split()[0]); print(sys.executable)"',
        timeout=10,
        strict_isolation=strict_isolation,
    )
    if code != 0:
        raise RuntimeError(f"Sandbox Python initialization failed (exit={code}): {out}")
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    sandbox_ver = lines[0] if lines else ""
    sandbox_exe = lines[1] if len(lines) > 1 else ""
    host_ver = sys.version.split()[0]
    if sandbox_ver != host_ver:
        raise RuntimeError(
            f"Sandbox Python version mismatch: sandbox={sandbox_ver}, host={host_ver}"
        )
    return {
        "sandbox_python_version": sandbox_ver,
        "sandbox_python_executable": sandbox_exe,
        "host_python_version": host_ver,
        "host_python_executable": sys.executable,
    }


def git_snapshot(workdir: str) -> dict[str, Any]:
    """Capture sha and worktree status."""

    def _git(*args: str) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", workdir, *args],
                capture_output=True,
                text=True,
                check=False,
            )
            return r.stdout.strip()
        except Exception:
            return ""

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    untracked: list[str] = []
    modified: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        code = line[:2]
        path = line[3:].strip()
        if code == "??":
            untracked.append(path)
        else:
            modified.append(path)
    return {
        "head": head,
        "status": status,
        "untracked": untracked,
        "modified": modified,
    }


SKIP_DIRS = {".git", "_hidden", ".bench_out", "score", "test-out"}


def snapshot(root: str) -> dict[str, str]:
    """relpath -> sha256 of regular files only. Symlinks are never followed."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d))
        )
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            out[os.path.relpath(path, root)] = digest.hexdigest()
    return out


def clean_copy(src: str, dst: str) -> None:
    """Rebuild a workspace from regular files only, so symlinks and special
    files created by the agent can never reach the scorer."""
    os.makedirs(dst, exist_ok=True)
    for rel in snapshot(src):
        target = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(target) or dst, exist_ok=True)
        shutil.copy2(os.path.join(src, rel), target)


def load_task(task_id: str) -> dict[str, Any]:
    """Load task metadata and file structure."""
    tdir = os.path.join(TASKS, task_id)
    with open(os.path.join(tdir, "meta.json")) as f:
        meta = json.load(f)
    meta["id"] = task_id
    meta["dir"] = tdir
    meta["task_dir"] = tdir
    meta["tests_dir"] = os.path.join(tdir, "tests")
    meta["repo_dir"] = os.path.join(tdir, "repo")
    meta["solution_dir"] = os.path.join(tdir, "solution")
    return meta


def all_tasks() -> list[str]:
    """Discover all available benchmark task IDs."""
    if not os.path.isdir(TASKS):
        return []
    ids = []
    for name in sorted(os.listdir(TASKS)):
        p = os.path.join(TASKS, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
            ids.append(name)
    return ids


def prepare(meta: dict[str, Any], run_dir: str) -> str:
    """Copy the starting repo into a disposable worktree and record a baseline."""
    work = os.path.join(run_dir, "work")
    if os.path.exists(work):
        shutil.rmtree(work)
    tdir = meta.get("task_dir") or meta.get("dir", "")
    shutil.copytree(os.path.join(tdir, "repo"), work)
    subprocess.run(["git", "init", "-q"], cwd=work, check=False)
    subprocess.run(["git", "add", "-A"], cwd=work, check=False)
    subprocess.run(
        ["git", "-c", "user.email=b@b", "-c", "user.name=b", "commit", "-qm", "base"],
        cwd=work,
        check=False,
    )
    with open(os.path.join(run_dir, "baseline.json"), "w") as f:
        json.dump(snapshot(work), f, indent=1)
    return work


def install_tests(meta: dict[str, Any], dest: str) -> bool:
    """Install hidden tests into a scorer-owned copy of the workspace."""
    tdir = meta.get("task_dir") or meta.get("dir", "")
    tests_src = os.path.join(tdir, "tests")
    if not os.path.isdir(tests_src):
        return False
    target = os.path.join(dest, "_hidden")
    if os.path.lexists(target):
        raise RuntimeError(f"scoring workspace already has {target!r}")
    shutil.copytree(tests_src, target)
    with open(os.path.join(target, "__init__.py"), "a"):
        pass
    return True


def score(
    meta: dict[str, Any],
    workdir: str,
    run_dir: str,
    strict_isolation: bool = False,
) -> tuple[bool, str]:
    """Score a clean rebuild of the agent's workspace, never the workspace."""
    scoring = os.path.join(run_dir, "score")
    if os.path.exists(scoring):
        shutil.rmtree(scoring)
    clean_copy(workdir, scoring)
    install_tests(meta, scoring)
    timeout = meta.get("timeout_s", meta.get("test_timeout_s", 120))
    code, out = run_sandboxed(
        scoring,
        meta["test_cmd"],
        timeout=timeout,
        outdir=os.path.join(run_dir, "test-out"),
        strict_isolation=strict_isolation,
    )
    with open(os.path.join(run_dir, "test_output.txt"), "w") as f:
        f.write(f"exit={code}\n{out}")
    return code == 0, out


def policy_check(meta: dict[str, Any], workdir: str, run_dir: str) -> dict[str, Any]:
    """Compare the workspace against the baseline taken before the agent ran."""
    baseline_path = os.path.join(run_dir, "baseline.json")
    baseline: dict[str, str] = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    current = snapshot(workdir)
    changed = sorted(k for k in current if k in baseline and current[k] != baseline[k])
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))

    special_or_symlinks: list[str] = []
    for dirpath, dirnames, filenames in os.walk(workdir, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if os.path.islink(p):
                special_or_symlinks.append(os.path.relpath(p, workdir))
        for name in filenames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p) or not os.path.isfile(p):
                special_or_symlinks.append(os.path.relpath(p, workdir))

    allowed = set(meta.get("allowed_files", []))
    forbidden = [p for p in (changed + added + removed) if allowed and p not in allowed]
    # Unconditionally reject new or altered special filesystem entries and symlinks
    forbidden.extend(special_or_symlinks)

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "forbidden": sorted(set(forbidden)),
        "policy_enforced": bool(allowed),
    }


def _mock_chat_stream(
    model: str,
    msgs: list[dict[str, Any]],
    ctx: int,
    ngpu: int,
    sampling: SamplingConfig | None = None,
    tools: list[dict[str, Any]] | None = None,
    timeout: int = 1800,
    url: str = "http://127.0.0.1:11434/api/chat",
    step: int = 0,
    request_idx: int = 1,
) -> StreamResult:
    """Deterministic mock model for integration testing without live Ollama."""
    task_prompt = msgs[1].get("content", "") if len(msgs) > 1 else ""
    is_tools = bool(tools)

    telemetry = {
        "step": step,
        "request_idx": request_idx,
        "ttft_s": 0.01,
        "ttft_thinking_s": None,
        "ttft_content_s": 0.01,
        "ttft_tool_s": 0.01 if is_tools else None,
        "prompt_eval_count": 40,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration_s": 0.01,
        "prompt_eval_tokens_per_s": 4000.0,
        "eval_count": 30,
        "eval_duration_s": 0.02,
        "decode_tokens_per_s": 1500.0,
        "total_duration_s": 0.03,
        "load_duration_s": 0.0,
        "transport_wall_s": 0.03,
        "done_reason": "stop",
        "content_length_chars": 50,
        "thinking_length_chars": 0,
        "tool_call_count": 1 if is_tools and step == 0 else 0,
    }

    if step == 0:
        if "cache.py" in task_prompt:
            sol_path = os.path.join(
                ROOT, "bench", "tasks", "debugging-default-arg", "solution", "cache.py"
            )
            with open(sol_path) as sf:
                sol = sf.read().strip()
            cmd = f'cat > cache.py << "EOF"\n{sol}\nEOF'
        elif "chunker.py" in task_prompt:
            sol_path = os.path.join(
                ROOT, "bench", "tasks", "debugging-off-by-one", "solution", "chunker.py"
            )
            with open(sol_path) as sf:
                sol = sf.read().strip()
            cmd = f'cat > chunker.py << "EOF"\n{sol}\nEOF'
        else:
            cmd = "echo 'mock fix'"

        if is_tools:
            return StreamResult(
                content="",
                thinking="",
                tool_calls=[
                    {
                        "id": "mock_call_1",
                        "function": {
                            "name": "execute_command",
                            "arguments": {"command": cmd},
                        },
                    }
                ],
                telemetry=telemetry,
            )
        return StreamResult(
            content=f"RUN: {cmd}",
            thinking="",
            tool_calls=[],
            telemetry=telemetry,
        )

    if is_tools:
        return StreamResult(
            content="",
            thinking="",
            tool_calls=[
                {
                    "id": "mock_call_2",
                    "function": {"name": "finish_task", "arguments": {}},
                }
            ],
            telemetry=telemetry,
        )
    return StreamResult(
        content="DONE",
        thinking="",
        tool_calls=[],
        telemetry=telemetry,
    )


def run_task(
    meta: dict[str, Any],
    model: str,
    ctx: int,
    ngpu: int,
    max_steps: int,
    max_wall: int,
    run_dir: str,
    sampling: SamplingConfig | None = None,
    actuator: str = "text",
    verbose: bool = True,
    chat_fn: Any = None,
    strict_isolation: bool = False,
) -> dict[str, Any]:
    """Execute benchmark task trajectory with full telemetry and sandbox isolation."""
    os.makedirs(run_dir, exist_ok=True)
    work = prepare(meta, run_dir)
    digest = task_digest(meta["id"])
    prompt_text = meta.get("prompt") or meta.get("instruction", "")
    user = f"{prompt_text}\n\nWork in /mnt/work. The repo is already checked out."

    if sampling is None:
        sampling = SamplingConfig()

    sys_prompt = (
        SYSTEM_PROMPT_TOOLS.format(max_steps=max_steps)
        if actuator == "native_tools"
        else SYSTEM_PROMPT_TEXT.format(max_steps=max_steps)
    )

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]

    if chat_fn is None:
        if os.environ.get("BENCH_MOCK_MODEL") == "1":
            chat_fn = _mock_chat_stream
        else:
            chat_fn = chat_stream

    vram_mon = VramMonitor(interval_s=0.5)
    vram_mon.start()

    t0 = time.time()
    steps = 0
    model_turns = 0
    commands_executed = 0
    max_commands = max_steps
    tokens = 0
    tool_calls = 0
    invalid = 0
    timed_out = False
    telemetry_records: list[dict[str, Any]] = []
    request_count = 0
    step_records: list[dict[str, Any]] = []
    inference_error: str | None = None

    try:
        while model_turns < max_steps:
            rem = max_wall - (time.time() - t0)
            if rem <= 0:
                timed_out = True
                break

            request_count += 1
            tools_arg = BENCH_TOOLS if actuator == "native_tools" else None
            try:
                res: StreamResult = chat_fn(
                    model,
                    msgs,
                    ctx,
                    ngpu,
                    sampling=sampling,
                    tools=tools_arg,
                    timeout=min(1800, max(1, int(rem))),
                    step=model_turns,
                    request_idx=request_count,
                )
            except Exception as e:
                inference_error = str(e)
                break

            telemetry_records.append(res.telemetry)
            tokens += res.telemetry.get("eval_count", 0) + res.telemetry.get(
                "prompt_eval_count", 0
            )
            model_turns += 1
            steps = model_turns

            reply = res.content
            thinking = res.thinking
            tool_calls_list = res.tool_calls

            # Incomplete streams are rejected immediately and cannot trigger execution
            if res.telemetry.get("incomplete_stream"):
                inference_error = (
                    "Inference stream closed prematurely without done event"
                )
                step_entry = {
                    "step": steps,
                    "model_turns": model_turns,
                    "commands_executed": commands_executed,
                    "reply": reply,
                    "thinking": thinking,
                    "action": "incomplete_stream_rejected",
                    "telemetry": res.telemetry,
                }
                step_records.append(step_entry)
                transcript.append(step_entry)
                break

            # --- Actuator: Native Tools ---
            if actuator == "native_tools":
                if not tool_calls_list:
                    invalid += 1
                    step_entry = {
                        "step": steps,
                        "model_turns": model_turns,
                        "commands_executed": commands_executed,
                        "reply": reply,
                        "thinking": thinking,
                        "action": "invalid_no_tool_call",
                        "telemetry": res.telemetry,
                    }
                    step_records.append(step_entry)
                    transcript.append(step_entry)
                    assistant_msg = {"role": "assistant", "content": reply or ""}
                    if thinking:
                        assistant_msg["thinking"] = thinking
                    msgs.append(assistant_msg)
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "Error: You must invoke a tool "
                                "('execute_command' or 'finish_task'). "
                                "Plain text commands are not accepted in "
                                "native tool mode."
                            ),
                        }
                    )
                    continue

                # Ambiguous batching rule: finish_task cannot coexist with other calls
                has_finish = any(
                    c.get("function", {}).get("name") == "finish_task"
                    for c in tool_calls_list
                )
                if has_finish and len(tool_calls_list) > 1:
                    invalid += 1
                    err_msg = (
                        "Protocol error: 'finish_task' cannot be batched with other "
                        "tool calls. Batch rejected."
                    )
                    step_entry = {
                        "step": steps,
                        "model_turns": model_turns,
                        "commands_executed": commands_executed,
                        "action": "invalid_mixed_finish_batch",
                        "tool_calls": tool_calls_list,
                        "telemetry": res.telemetry,
                    }
                    step_records.append(step_entry)
                    transcript.append(step_entry)
                    assistant_msg = {
                        "role": "assistant",
                        "content": reply or "",
                        "tool_calls": tool_calls_list,
                    }
                    if thinking:
                        assistant_msg["thinking"] = thinking
                    msgs.append(assistant_msg)
                    for c_idx, c in enumerate(tool_calls_list):
                        c_id = c.get("id", f"call_{steps}_{c_idx}")
                        msgs.append(
                            {
                                "role": "tool",
                                "content": err_msg,
                                "tool_name": c.get("function", {}).get("name", "tool"),
                                "tool_call_id": c_id,
                            }
                        )
                    continue

                if has_finish and len(tool_calls_list) == 1:
                    call = tool_calls_list[0]
                    call_id = call.get("id", f"call_{steps}_0")
                    tool_calls += 1
                    step_record = {
                        "step": steps,
                        "model_turns": model_turns,
                        "commands_executed": commands_executed,
                        "action": "done",
                        "tool_call": call,
                        "telemetry": res.telemetry,
                    }
                    step_records.append(step_record)
                    transcript.append(step_record)
                    assistant_msg = {
                        "role": "assistant",
                        "content": reply or "",
                        "tool_calls": tool_calls_list,
                    }
                    if thinking:
                        assistant_msg["thinking"] = thinking
                    msgs.append(assistant_msg)
                    msgs.append(
                        {
                            "role": "tool",
                            "content": "Task completed.",
                            "tool_name": "finish_task",
                            "tool_call_id": call_id,
                        }
                    )
                    break

                assistant_msg = {
                    "role": "assistant",
                    "content": reply or "",
                    "tool_calls": tool_calls_list,
                }
                if thinking:
                    assistant_msg["thinking"] = thinking
                msgs.append(assistant_msg)

                batch_timed_out = False
                for call_idx, call in enumerate(tool_calls_list):
                    # Check remaining wall-clock budget before each command in batch
                    rem_cmd = max_wall - (time.time() - t0)
                    if rem_cmd <= 0:
                        timed_out = True
                        batch_timed_out = True
                        break

                    call_id = call.get("id", f"call_{steps}_{call_idx}")
                    fn = call.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {}
                    if not isinstance(fn_args, dict):
                        fn_args = {}

                    tool_calls += 1

                    if fn_name == "execute_command":
                        cmd = fn_args.get("command")
                        if not isinstance(cmd, str) or not cmd.strip():
                            invalid += 1
                            err_msg = (
                                "Error: 'command' argument must be a non-empty string."
                            )
                            step_record = {
                                "step": steps,
                                "model_turns": model_turns,
                                "commands_executed": commands_executed,
                                "tool_call": call,
                                "invalid_argument": True,
                                "telemetry": res.telemetry,
                            }
                            step_records.append(step_record)
                            transcript.append(step_record)
                            msgs.append(
                                {
                                    "role": "tool",
                                    "content": err_msg,
                                    "tool_name": fn_name,
                                    "tool_call_id": call_id,
                                }
                            )
                        elif commands_executed >= max_commands:
                            invalid += 1
                            err_msg = (
                                f"Error: Shell command budget exhausted "
                                f"({max_commands} commands limit). "
                                "No more commands may be executed. "
                                "Call 'finish_task' to conclude."
                            )
                            step_record = {
                                "step": steps,
                                "model_turns": model_turns,
                                "commands_executed": commands_executed,
                                "tool_call": call,
                                "command_budget_exhausted": True,
                                "telemetry": res.telemetry,
                            }
                            step_records.append(step_record)
                            transcript.append(step_record)
                            msgs.append(
                                {
                                    "role": "tool",
                                    "content": err_msg,
                                    "tool_name": fn_name,
                                    "tool_call_id": call_id,
                                }
                            )
                        else:
                            commands_executed += 1
                            budget = max(
                                1,
                                min(
                                    meta.get("cmd_timeout_s", 60),
                                    int(rem_cmd),
                                ),
                            )
                            code, out = run_sandboxed(
                                work,
                                cmd,
                                budget,
                                os.path.join(run_dir, "cmd-out"),
                                strict_isolation=strict_isolation,
                            )
                            feed = f"[exit={code}]\n{out}" if out else f"[exit={code}]"
                            step_record = {
                                "step": steps,
                                "model_turns": model_turns,
                                "commands_executed": commands_executed,
                                "command": cmd,
                                "code": code,
                                "output": out,
                                "tool_call": call,
                                "telemetry": res.telemetry,
                            }
                            step_records.append(step_record)
                            transcript.append(step_record)
                            msgs.append(
                                {
                                    "role": "tool",
                                    "content": feed,
                                    "tool_name": fn_name,
                                    "tool_call_id": call_id,
                                }
                            )
                    else:
                        invalid += 1
                        err_msg = (
                            f"Error: Unknown tool '{fn_name}'. Available tools are "
                            "'execute_command' and 'finish_task'."
                        )
                        step_record = {
                            "step": steps,
                            "model_turns": model_turns,
                            "commands_executed": commands_executed,
                            "tool_call": call,
                            "unknown_tool": True,
                            "telemetry": res.telemetry,
                        }
                        step_records.append(step_record)
                        transcript.append(step_record)
                        msgs.append(
                            {
                                "role": "tool",
                                "content": err_msg,
                                "tool_name": fn_name,
                                "tool_call_id": call_id,
                            }
                        )

                if batch_timed_out:
                    break
                continue

            # --- Actuator: Text (RUN: / DONE) ---
            kind, cmd = parse_action(reply)
            if kind is None and thinking and not reply.strip():
                rem_retry = max(1, int(max_wall - (time.time() - t0)))
                retry_msgs = [
                    *msgs,
                    {"role": "assistant", "content": thinking},
                    {
                        "role": "user",
                        "content": (
                            "You did not output an action. Emit exactly one command "
                            "starting with 'RUN: ' or the single word 'DONE'."
                        ),
                    },
                ]
                res_retry = chat_fn(
                    model,
                    retry_msgs,
                    ctx,
                    ngpu,
                    sampling=sampling,
                    timeout=min(1800, rem_retry),
                    step=steps,
                    request_idx=request_count,
                )
                telemetry_records.append(res_retry.telemetry)
                tokens += res_retry.telemetry.get(
                    "eval_count", 0
                ) + res_retry.telemetry.get("prompt_eval_count", 0)
                reply = res_retry.content
                kind, cmd = parse_action(reply)

            step_entry: dict[str, Any] = {
                "step": steps,
                "model_turns": model_turns,
                "commands_executed": commands_executed,
                "reply": reply,
                "thinking": thinking,
                "action": kind,
                "telemetry": res.telemetry,
            }
            if kind == "done":
                transcript.append(step_entry)
                break
            if kind != "run" or not cmd:
                invalid += 1
                transcript.append(step_entry)
                msgs.append({"role": "assistant", "content": reply})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "Invalid action. Emit either 'RUN: <command>' or 'DONE'."
                        ),
                    }
                )
                continue

            tool_calls += 1
            rem_cmd = max_wall - (time.time() - t0)
            if rem_cmd <= 0:
                timed_out = True
                break

            if commands_executed >= max_commands:
                invalid += 1
                step_entry["command_budget_exhausted"] = True
                transcript.append(step_entry)
                msgs.append({"role": "assistant", "content": reply})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Error: Shell command budget exhausted "
                            f"({max_commands} commands limit). "
                            "Output 'DONE' to finish."
                        ),
                    }
                )
                continue

            commands_executed += 1
            budget = max(
                1,
                min(
                    meta.get("cmd_timeout_s", 60),
                    int(rem_cmd),
                ),
            )
            code, out = run_sandboxed(
                work,
                cmd,
                budget,
                os.path.join(run_dir, "cmd-out"),
                strict_isolation=strict_isolation,
            )
            feed = f"[exit={code}]\n{out}" if out else f"[exit={code}]"
            step_entry["command"] = cmd
            step_entry["code"] = code
            step_entry["output"] = out
            step_records.append(step_entry)
            transcript.append(step_entry)
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": feed})

    finally:
        vram_data = vram_mon.stop()

    t_agent_end = time.time()
    agent_wall_s = t_agent_end - t0
    t_verify_start = time.time()
    tests_ok, test_out = score(meta, work, run_dir, strict_isolation=strict_isolation)
    diff = policy_check(meta, work, run_dir)
    verification_wall_s = time.time() - t_verify_start
    total_wall_s = time.time() - t0
    passed = tests_ok and not diff["forbidden"]

    infer_summary = summarize_inference(telemetry_records)
    offload_info = get_ollama_offload_status(model)

    classification = classify_task_run(
        passed=passed,
        timed_out=timed_out,
        step_count=steps,
        max_steps=max_steps,
        invalid_actions=invalid,
        forbidden_files=diff["forbidden"],
        changed_files=diff["changed"],
        exit_status=test_out.splitlines()[0] if test_out else "none",
        steps=step_records,
        num_ctx=ctx,
        telemetry_records=telemetry_records,
        inference_error=inference_error,
    )

    with open(os.path.join(run_dir, "transcript.json"), "w") as f:
        json.dump(
            {
                "task": meta["id"],
                "task_digest": digest,
                "actuator": actuator,
                "sampling": sampling.to_dict(),
                "steps": transcript,
                "inference_telemetry": telemetry_records,
                "classification": classification,
            },
            f,
            indent=1,
        )
    with open(os.path.join(run_dir, "test_output.txt"), "w") as f:
        f.write(test_out)

    rec = {
        "task": meta["id"],
        "task_digest": digest,
        "passed": passed,
        "outcome": classification["outcome"],
        "possible_cause": classification["possible_cause"],
        "cause_evidence": classification["evidence"],
        "symptoms": classification.get("symptoms", {}),
        "strict_isolation": strict_isolation,
        "actuator": actuator,
        "sampling": sampling.to_dict(),
        "steps": model_turns,
        "model_turns": model_turns,
        "commands_executed": commands_executed,
        "max_commands": max_commands,
        "tokens": tokens,
        "agent_wall_s": round(agent_wall_s, 2),
        "verification_wall_s": round(verification_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
        "wall_s": round(total_wall_s, 2),
        "peak_vram_mib": vram_data.get("peak_device_vram_mib"),
        "peak_device_vram_mib": vram_data.get("peak_device_vram_mib"),
        "peak_ollama_vram_mib": vram_data.get("peak_ollama_vram_mib"),
        "gpu_telemetry_status": vram_data.get("telemetry_status"),
        "monitor_sample_count": vram_data.get("monitor_sample_count"),
        "monitor_max_sample_gap_s": vram_data.get("monitor_max_sample_gap_s"),
        "monitor_overhead_s": vram_data.get("monitor_overhead_s"),
        "reported_vram_size_ratio": offload_info.get("reported_vram_size_ratio"),
        "runner": offload_info.get("runner"),
        "tool_calls": tool_calls,
        "invalid_actions": invalid,
        "timed_out": timed_out,
        "changed_files": diff["changed"],
        "added_files": diff["added"],
        "removed_files": diff["removed"],
        "forbidden_files": diff["forbidden"],
        "inference_summary": infer_summary,
        "exit": test_out.splitlines()[0] if test_out else "",
    }
    if verbose:
        timing = (
            f"{rec['wall_s']:6.1f}s "
            f"(agent: {agent_wall_s:5.1f}s, verify: {verification_wall_s:4.1f}s)"
        )
        cause_suffix = (
            f" [{classification['possible_cause']}]"
            if classification["possible_cause"]
            else ""
        )
        fail_desc = (
            f" {classification['outcome']}{cause_suffix}"
            if not tests_ok or diff["forbidden"]
            else ""
        )
        print(
            f"  {'PASS' if passed else 'FAIL'}  {meta['id']:24s} "
            f"{timing} "
            f"{rec['steps']:2d} steps {rec['tokens']:6d} tok "
            f"{'REGRESSION' if diff['forbidden'] else ''}"
            f"{' TIMEOUT' if timed_out else ''}"
            f"{fail_desc}",
            flush=True,
        )
    return rec


# ---------- environment & provenance ----------


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def environment() -> dict[str, Any]:
    """Capture system and execution environment details for reproducibility."""
    raw_env = _sh("systemctl --user show ollama --property=Environment --value")
    safe_ollama_env = {}
    safe_vars = {
        "OLLAMA_FLASH_ATTENTION",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_KEEP_ALIVE",
        "OLLAMA_HOST",
        "OLLAMA_MODELS",
    }
    for item in raw_env.split():
        if "=" in item:
            k, v = item.split("=", 1)
            if k in safe_vars:
                if "/" in v and "home" in v:
                    safe_ollama_env[k] = "<user_path>"
                else:
                    safe_ollama_env[k] = v

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama": _sh("ollama --version"),
        "nvidia_driver": _sh(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
        ),
        "gpu": _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
        "power_limit_w": _sh(
            "nvidia-smi --query-gpu=power.limit --format=csv,noheader"
        ),
        "python": sys.version.split()[0],
        "cpu": _sh("nproc"),
        "ram_gib": _sh("free -g | awk '/^Mem:/{print $2}'"),
        "ollama_env": safe_ollama_env,
        "models": _sh("ollama list | awk '{print $1, $2, $3, $4}'"),
        "isolation_backend": "cgroup_v2" if check_cgroup_v2() else "posix_fallback",
        "gpu_telemetry": check_gpu_telemetry(),
    }


def model_fingerprint(model: str) -> dict[str, Any]:
    """Full manifest fingerprint: config digest, weights digest, and size."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    root = os.path.expanduser("~/.ollama/models/manifests")
    want = os.sep + os.path.join(*name.split("/"))
    for dirpath, _, files in os.walk(root):
        if dirpath.endswith(want) and tag in files:
            try:
                with open(os.path.join(dirpath, tag)) as f:
                    data = json.load(f)
                config_d = data.get("config", {}).get("digest")
                model_layer = next(
                    (
                        layer
                        for layer in data.get("layers", [])
                        if "model" in layer.get("mediaType", "")
                    ),
                    {},
                )
                return {
                    "config_digest": config_d,
                    "weights_digest": model_layer.get("digest"),
                    "weights_bytes": model_layer.get("size"),
                }
            except Exception:
                break
    listed = _sh(f"ollama list | awk '$1==\"{model}\"{{print $2}}'")
    return {
        "config_digest": listed + " (abbreviated)",
        "weights_digest": None,
        "weights_bytes": None,
    }


def model_digest(model: str) -> str:
    """Convenience wrapper returning weights digest or config digest."""
    fp = model_fingerprint(model)
    return str(fp["weights_digest"] or fp["config_digest"])


def _resolve_run_dir_and_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[str, dict[str, Any], SamplingConfig]:
    sampling = SamplingConfig(
        profile=args.sampling_profile,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )

    if args.resume:
        run_dir = os.path.abspath(args.resume)
        if not os.path.isdir(run_dir):
            print(f"Error: resume directory does not exist: {run_dir}", file=sys.stderr)
            sys.exit(1)
        config_path = os.path.join(run_dir, "config.json")
        saved: dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                saved = json.load(f)
            if (
                args.model
                and args.model != saved.get("model")
                and args.model != parser.get_default("model")
            ):
                expected = saved.get("model")
                print(
                    f"Error: model mismatch on resume (run has {expected}, "
                    f"requested {args.model})",
                    file=sys.stderr,
                )
                sys.exit(1)
        if "sampling" in saved and isinstance(saved["sampling"], dict):
            sampling = SamplingConfig.from_dict(saved["sampling"])

        cfg = {
            "model": saved.get("model", args.model),
            "ctx": saved.get("ctx", args.ctx),
            "ngpu": saved.get("ngpu", args.ngpu),
            "steps": saved.get("steps", args.steps),
            "repeat": saved.get("repeat", args.repeat),
            "max_wall": saved.get("max_wall", args.max_wall),
            "actuator": saved.get("actuator", args.actuator),
            "strict_isolation": saved.get("strict_isolation", False)
            or args.strict_isolation,
        }
        return run_dir, cfg, sampling

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"-{args.tag}" if args.tag else ""
        run_dir = os.path.join(RUNS, f"{stamp}{tag}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = {
        "model": args.model,
        "ctx": args.ctx,
        "ngpu": args.ngpu,
        "steps": args.steps,
        "repeat": args.repeat,
        "max_wall": args.max_wall,
        "tag": args.tag or "",
        "task": args.task or "",
        "actuator": args.actuator,
        "strict_isolation": args.strict_isolation,
        "sampling": sampling.to_dict(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return run_dir, cfg, sampling


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--ngpu", type=int, default=99)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--task", help="run a single task by ID")
    p.add_argument("--tag", help="label for the run output directory")
    p.add_argument(
        "--max-wall",
        type=int,
        default=1200,
        help="task wall-clock budget in seconds",
    )
    p.add_argument(
        "--actuator",
        choices=["text", "native_tools"],
        default="text",
        help="actuator mechanism: text (RUN: / DONE) or native_tools",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-profile", default="deterministic-v1")
    p.add_argument(
        "--strict-isolation",
        action="store_true",
        help="fail closed if cgroup v2 hard memory limits are unavailable",
    )
    p.add_argument("--resume", help="path to existing run directory to resume")
    p.add_argument("--run-dir", help="custom destination directory for this run")
    args = p.parse_args()

    run_dir, cfg, sampling = _resolve_run_dir_and_config(args, p)

    if cfg.get("strict_isolation") and not check_cgroup_v2():
        print(
            "Error: --strict-isolation requested, but cgroup v2 is unavailable. "
            "Failing closed.",
            file=sys.stderr,
        )
        return 1

    ids = [args.task] if args.task else all_tasks()
    if not ids:
        print("no tasks found", file=sys.stderr)
        return 1

    model = cfg["model"]
    ctx = cfg["ctx"]
    ngpu = cfg["ngpu"]
    max_steps = cfg["steps"]
    repeat = cfg["repeat"]
    max_wall = cfg["max_wall"]
    actuator = cfg.get("actuator", "text")

    init_check_dir = os.path.join(run_dir, ".init-check")
    os.makedirs(init_check_dir, exist_ok=True)
    try:
        py_info = assert_sandbox_python(
            init_check_dir, strict_isolation=cfg.get("strict_isolation", False)
        )
    finally:
        shutil.rmtree(init_check_dir, ignore_errors=True)

    env = environment()
    fp = model_fingerprint(model)
    env["model_fingerprint"] = fp
    env["model_digest"] = fp["weights_digest"] or fp["config_digest"]
    env["strict_isolation"] = cfg.get("strict_isolation", False)
    env.update(py_info)
    env["sandbox_python"] = py_info["sandbox_python_version"]
    env["host_python_exact"] = sys.version

    env_path = os.path.join(run_dir, "env.json")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)

    iso = env.get("isolation_backend", "posix_fallback")
    print(
        f"RUN {run_dir}\n"
        f"model={model} ({env['model_digest']})  "
        f"ctx={ctx}  ngpu={ngpu}  tasks={len(ids)}  repeat={repeat}\n"
        f"actuator={actuator}  sampling={sampling.profile} "
        f"(temp={sampling.temperature})\n"
        f"host={env['gpu']} (driver {env['nvidia_driver']})  "
        f"python={env['python']} (sandbox={env['sandbox_python']})  "
        f"isolation={iso}\n",
        flush=True,
    )

    results_path = os.path.join(run_dir, "results.json")
    recs: list[dict[str, Any]] = []
    completed_keys: set[tuple[int, str]] = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                recs = json.load(f)
                completed_keys = {
                    (r.get("repeat", 0), r.get("task")) for r in recs if r.get("task")
                }
                print(f"Resuming run: found {len(completed_keys)} completed task(s).")
        except Exception:
            recs = []

    for rep in range(repeat):
        for tid in ids:
            if (rep, tid) in completed_keys:
                continue
            meta = load_task(tid)
            tdir = os.path.join(run_dir, f"r{rep}", tid)
            if os.path.exists(tdir):
                shutil.rmtree(tdir, ignore_errors=True)
            rec = run_task(
                meta,
                model,
                ctx,
                ngpu,
                max_steps,
                max_wall,
                tdir,
                sampling=sampling,
                actuator=actuator,
                strict_isolation=cfg.get("strict_isolation", False),
            )
            rec["task"] = tid
            rec["repeat"] = rep
            recs.append(rec)
            tmp_results = results_path + f".tmp.{os.getpid()}"
            with open(tmp_results, "w") as f:
                json.dump(recs, f, indent=2)
            os.replace(tmp_results, results_path)

    ok = sum(1 for r in recs if r["passed"])
    n = len(recs)
    total_h = sum(r["total_wall_s"] for r in recs) / 3600 or 1e-9
    agent_h = sum(r["agent_wall_s"] for r in recs) / 3600 or 1e-9
    verify_h = sum(r["verification_wall_s"] for r in recs) / 3600 or 1e-9
    vram_vals = [
        int(r["peak_vram_mib"]) for r in recs if r.get("peak_vram_mib") is not None
    ]
    peak_vram_str = f"{max(vram_vals)} MiB" if vram_vals else "n/a"
    print(
        f"\n{n} runs  pass={ok}/{n} ({100 * ok / n:.0f}%)  "
        f"verified tasks/hour={ok / total_h:.2f}  "
        f"(agent: {agent_h * 3600:.1f}s, verify: {verify_h * 3600:.1f}s)  "
        f"peak_vram={peak_vram_str}  "
        f"invalid_actions={sum(r.get('invalid_actions', 0) for r in recs)}"
    )
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
