#!/usr/bin/env python3
"""Agentic coding benchmark for local models on a fixed VRAM budget.

Runs a shell-only agent loop (one command per turn, mini-SWE-agent style)
against disposable task repos, scores with hidden tests, records metrics.

Usage:
    python3 bench.py --model qwen3.5:4b --ctx 65536 --ngpu 99
    python3 bench.py --model qwen3.5:4b --ctx 65536 --tasks debug-tz-offset
    python3 bench.py --model qwen3.5:4b --list
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

# uv-managed CPython, bound read-only into the sandbox as /opt/python.
# The sandbox masks /home, so a host interpreter under $HOME is invisible
# unless it is bound explicitly.
PYTHON_HOME = os.environ.get(
    "BENCH_PYTHON_HOME",
    "/home/alxnko/.local/share/uv/python/cpython-3.15.2-linux-x86_64-gnu",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(ROOT, "tasks")
RUNS = os.path.join(ROOT, "runs")

SYSTEM = """You are a coding agent working in a git repository at /mnt/work.

You act by running shell commands. Reply with exactly one action per turn.

To run a command:
RUN: <the command>

When the task is complete:
DONE

Rules:
- One command per turn. Wait for its output before the next.
- You have {max_steps} turns. Read files with cat/head/tail, edit with
  python3 - <<'EOF' heredocs or cat > file <<'EOF', inspect with ls/grep.
- Tests are hidden. Make the code correct, not the tests pass.
- No network access. No package installs.
"""


# ---------- ollama ----------


def chat(model, messages, ctx, ngpu, num_predict=2048, timeout=1800):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": ctx,
            "num_gpu": ngpu,
            "num_predict": num_predict,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(d["error"])
    msg = d["message"]
    return msg.get("content", ""), msg.get("thinking", ""), d.get("eval_count", 0)


def unload(model):
    payload = {"model": model, "keep_alive": 0}
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


# ---------- sandbox ----------

# Only the toolchain is mounted. Binding "/" read-only would still expose every
# readable file on the machine (ssh keys, tokens, browser profiles).
ETC_FILES = [
    "/etc/passwd",
    "/etc/group",
    "/etc/nsswitch.conf",
    "/etc/hosts",
    "/etc/hostname",
    "/etc/resolv.conf",
    "/etc/localtime",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/ssl/certs",
]

OUTPUT_CAP = 4096  # bytes of command output shown to the model


def bwrap_cmd(workdir, inner, outdir=None):
    """Filesystem + network isolation with a curated, read-only toolchain."""
    args = ["bwrap", "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib64"):
        host = "/" + name
        if os.path.islink(host):
            args += ["--symlink", os.readlink(host), host]
        elif os.path.isdir(host):
            args += ["--ro-bind", host, host]
    for path in ETC_FILES:
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += [
        "--ro-bind",
        PYTHON_HOME,
        "/opt/python",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/root",
        "--tmpfs",
        "/mnt",
        "--bind",
        os.path.abspath(workdir),
        "/mnt/work",
    ]
    if outdir:
        args += ["--bind", os.path.abspath(outdir), "/mnt/out"]
    args += [
        "--chdir",
        "/mnt/work",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--new-session",
        "--",
        "bash",
        "-c",
        inner,
    ]
    return args


def run_sandboxed(workdir, inner, timeout, outdir=None):
    """Run under bwrap. Output is written to disk in the sandbox and only its
    tail is read back, so a runaway command cannot exhaust host memory."""
    outdir = outdir or tempfile.mkdtemp(prefix="bench-out-")
    os.makedirs(outdir, exist_ok=True)
    stdout_path = os.path.join(outdir, "stdout")
    rc_path = os.path.join(outdir, "rc")
    for path in (stdout_path, rc_path):
        if os.path.exists(path):
            os.remove(path)
    # -t cpu seconds, -f file size, -u process count
    wrapped = (
        "ulimit -t 120 2>/dev/null; ulimit -f 16384 2>/dev/null; "
        "ulimit -u 512 2>/dev/null; "
        f"( {{ {inner} ; }} ) > /mnt/out/stdout 2>&1; echo $? > /mnt/out/rc"
    )
    try:
        proc = subprocess.run(
            bwrap_cmd(workdir, wrapped, outdir),
            capture_output=True,
            timeout=timeout,
            env={
                "PATH": "/opt/python/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
            },
        )
    except subprocess.TimeoutExpired:
        return 124, f"<timeout after {timeout}s>"
    try:
        with open(rc_path) as f:
            code = int(f.read().strip())
    except OSError, ValueError:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        msg = (
            f"<bwrap failed ({proc.returncode}): {err}>"
            if err
            else "<killed before reporting an exit code>"
        )
        return proc.returncode or 124, msg
    text = ""
    try:
        size = os.path.getsize(stdout_path)
        with open(stdout_path, "rb") as f:
            if size > OUTPUT_CAP:
                f.seek(size - OUTPUT_CAP)
                text = "...<output truncated>...\n"
            text += f.read().decode("utf-8", "replace")
    except OSError:
        pass
    return code, text


# ---------- vram sampler ----------


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self.stop.wait(0.5)


# ---------- task loading ----------


def load_task(task_id):
    d = os.path.join(TASKS, task_id)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    meta["id"] = task_id
    meta["dir"] = d
    return meta


def all_tasks():
    if not os.path.isdir(TASKS):
        return []
    return sorted(
        t
        for t in os.listdir(TASKS)
        if os.path.isfile(os.path.join(TASKS, t, "meta.json"))
    )


SKIP_DIRS = {".git", "_hidden"}


def snapshot(root):
    """relpath -> sha256 of regular files only. Symlinks are never followed."""
    out = {}
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


def clean_copy(src, dst):
    """Rebuild a workspace from regular files only, so symlinks and special
    files created by the agent can never reach the scorer."""
    os.makedirs(dst, exist_ok=True)
    for rel in snapshot(src):
        target = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(target) or dst, exist_ok=True)
        shutil.copy2(os.path.join(src, rel), target)


def prepare(meta, run_dir):
    """Copy the starting repo into a disposable worktree and record a baseline."""
    work = os.path.join(run_dir, "work")
    shutil.copytree(os.path.join(meta["dir"], "repo"), work)
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


def install_tests(meta, dest):
    """Install hidden tests into a scorer-owned copy of the workspace."""
    tests_src = os.path.join(meta["dir"], "tests")
    if not os.path.isdir(tests_src):
        return False
    target = os.path.join(dest, "_hidden")
    if os.path.lexists(target):
        raise RuntimeError(f"scoring workspace already has {target!r}")
    shutil.copytree(tests_src, target)
    open(os.path.join(target, "__init__.py"), "a").close()
    return True


def score(meta, work, run_dir):
    """Score a clean rebuild of the agent's workspace, never the workspace."""
    scoring = os.path.join(run_dir, "score")
    if os.path.exists(scoring):
        shutil.rmtree(scoring)
    clean_copy(work, scoring)
    install_tests(meta, scoring)
    code, out = run_sandboxed(
        scoring,
        meta["test_cmd"],
        meta.get("timeout_s", 120),
        os.path.join(run_dir, "test-out"),
    )
    with open(os.path.join(run_dir, "test_output.txt"), "w") as f:
        f.write(f"exit={code}\n{out}")
    return code == 0, out


def policy_check(meta, work, run_dir):
    """Compare the workspace against the baseline taken before the agent ran.
    Independent of git, so tampering with .git changes nothing."""
    with open(os.path.join(run_dir, "baseline.json")) as f:
        baseline = json.load(f)
    current = snapshot(work)
    changed = sorted(k for k in current if k in baseline and current[k] != baseline[k])
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    touched = changed + added + removed
    allowed = set(meta.get("allowed_files", []))
    forbidden = [p for p in touched if allowed and p not in allowed]
    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "forbidden": forbidden,
        "policy_enforced": bool(allowed),
    }


# ---------- agent loop ----------


def parse_action(text):
    """Return (kind, command). Multi-line commands (heredocs) are kept whole."""
    if not text:
        return None, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.upper().startswith("RUN:"):
            head = s[4:].strip()
            return "run", strip_fence("\n".join([head, *lines[i + 1 :]])).strip()
        if s.upper().startswith("DONE"):
            return "done", None
    # tolerate the XML-ish action wrappers some chat templates emit
    m = re.search(r"<(result|command|cmd|run|bash)>\s*(.+?)\s*</\1>", text, re.S | re.I)
    if m:
        return "run", strip_fence(m.group(2)).strip()
    # tolerate a bare fenced bash block as a command
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            body = blocks[i]
            body = body.split("\n", 1)[1] if "\n" in body else body
            if body.strip():
                return "run", body.strip()
    return None, None


def strip_fence(cmd):
    """Drop a trailing ``` fence that closed the command block."""
    lines = cmd.rstrip().splitlines()
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines)


def run_task(meta, model, ctx, ngpu, max_steps, run_root, verbose=True, max_wall=1200):
    run_dir = os.path.join(run_root, meta["id"])
    os.makedirs(run_dir, exist_ok=True)
    work = prepare(meta, run_dir)

    listing = subprocess.run(
        ["bash", "-c", "ls -R . | head -60"], cwd=work, capture_output=True, text=True
    ).stdout
    user = (
        f"{meta['prompt']}\n\nRepository layout:\n{listing}\n"
        f"You have {max_steps} turns. Start by inspecting the files."
    )

    msgs = [
        {"role": "system", "content": SYSTEM.format(max_steps=max_steps)},
        {"role": "user", "content": user},
    ]

    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    timed_out = False
    steps = 0
    tokens = 0
    invalid = 0
    tool_calls = 0
    transcript = []

    try:
        for step in range(max_steps):
            remaining = max_wall - (time.time() - t0)
            if remaining <= 1:
                timed_out = True
                break
            reply, thinking, ntok = chat(
                model, msgs, ctx, ngpu, timeout=min(1800, int(remaining))
            )
            tokens += ntok
            steps += 1
            kind, cmd = parse_action(reply)
            # Thinking models can spend the whole budget reasoning and emit no
            # content. Ask again instead of burning a step on empty output.
            if kind is None and thinking and not reply.strip():
                retry_msgs = [
                    *msgs,
                    {"role": "assistant", "content": thinking},
                    {
                        "role": "user",
                        "content": "Now write the action. Reply with exactly one line "
                        "starting with 'RUN: ' or the single word 'DONE'.",
                    },
                ]
                reply, _, ntok2 = chat(model, retry_msgs, ctx, ngpu)
                tokens += ntok2
                kind, cmd = parse_action(reply)
            transcript.append(
                {
                    "step": step,
                    "reply": reply,
                    "thinking": thinking[:4000],
                    "action": kind,
                    "cmd": cmd,
                }
            )
            if kind == "done":
                break
            if kind != "run" or not cmd:
                invalid += 1
                msgs.append({"role": "assistant", "content": reply or "(empty)"})
                msgs.append(
                    {
                        "role": "user",
                        "content": "OBSERVATION: no action found. Reply with "
                        "'RUN: <command>' or 'DONE'.",
                    }
                )
                if invalid >= 3:
                    break
                continue
            tool_calls += 1
            budget = max(
                1,
                min(meta.get("cmd_timeout_s", 60), int(max_wall - (time.time() - t0))),
            )
            code, out = run_sandboxed(
                work, cmd, budget, os.path.join(run_dir, "cmd-out")
            )
            msgs.append({"role": "assistant", "content": reply})
            msgs.append(
                {"role": "user", "content": f"OBSERVATION (exit {code}):\n{out}"}
            )
    finally:
        sampler.stop.set()
        sampler.join(timeout=3)

    elapsed = time.time() - t0
    tests_ok, test_out = score(meta, work, run_dir)
    diff = policy_check(meta, work, run_dir)
    passed = tests_ok and not diff["forbidden"]

    with open(os.path.join(run_dir, "transcript.json"), "w") as f:
        json.dump(transcript, f, indent=1)

    rec = {
        "task": meta["id"],
        "category": meta["category"],
        "model": model,
        "ctx": ctx,
        "ngpu": ngpu,
        "passed": passed,
        "steps": steps,
        "tokens": tokens,
        "wall_s": round(elapsed, 1),
        "peak_vram_mib": sampler.peak,
        "tool_calls": tool_calls,
        "invalid_actions": invalid,
        "tests_passed": tests_ok,
        "changed_files": diff["changed"],
        "added_files": diff["added"],
        "removed_files": diff["removed"],
        "unexpected_changes": diff["forbidden"],
        "policy_enforced": diff["policy_enforced"],
        "timed_out": timed_out,
        "exit": test_out.splitlines()[0] if test_out else "",
    }
    if verbose:
        print(
            f"  {'PASS' if passed else 'FAIL'}  {meta['id']:24s} "
            f"{rec['wall_s']:6.1f}s {rec['steps']:2d} steps "
            f"{rec['tokens']:6d} tok "
            f"{'REGRESSION' if diff['forbidden'] else ''}"
            f"{' TIMEOUT' if timed_out else ''}"
            f"{'' if tests_ok or diff['forbidden'] else ' tests-failed'}",
            flush=True,
        )
    return rec


def _sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        return ""


def environment():
    """Provenance: the numbers are meaningless without the versions behind them."""
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
        "ollama_unit": _sh(
            "systemctl --user show ollama --property=Environment --value"
        ),
        "models": _sh("ollama list"),
    }


def model_digest(model):
    """Full manifest digest. `ollama list` only prints a 12-char prefix."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    root = os.path.expanduser("~/.ollama/models/manifests")
    want = os.sep + os.path.join(*name.split("/"))
    for dirpath, _, files in os.walk(root):
        if dirpath.endswith(want) and tag in files:
            try:
                with open(os.path.join(dirpath, tag)) as f:
                    data = json.load(f)
                return data["config"]["digest"]
            except Exception:
                break
    listed = _sh(f"ollama list | awk '$1==\"{model}\"{{print $2}}'")
    return listed + " (abbreviated)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--ngpu", type=int, default=99)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument(
        "--max-wall",
        type=int,
        default=1200,
        help="per-task wall-clock budget in seconds",
    )
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ids = args.tasks or all_tasks()
    if args.list:
        for t in ids:
            m = load_task(t)
            print(f"{t:24s} {m['category']:12s} {m.get('title', '')}")
        return 0
    if not ids:
        print("no tasks found", file=sys.stderr)
        return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = os.path.join(RUNS, f"{stamp}-{args.tag or args.model.split('/')[-1]}")
    os.makedirs(run_root, exist_ok=True)

    env = environment()
    env["model_digest"] = model_digest(args.model)
    env["profile"] = {
        "model": args.model,
        "ctx": args.ctx,
        "ngpu": args.ngpu,
        "steps": args.steps,
        "repeat": args.repeat,
    }
    with open(os.path.join(run_root, "env.json"), "w") as f:
        json.dump(env, f, indent=1)

    print(
        f"model={args.model} ctx={args.ctx} ngpu={args.ngpu} "
        f"tasks={len(ids)} repeats={args.repeat}"
    )
    unload(args.model)
    time.sleep(2)

    recs = []
    for rep in range(args.repeat):
        for t in ids:
            recs.append(
                run_task(
                    load_task(t),
                    args.model,
                    args.ctx,
                    args.ngpu,
                    args.steps,
                    os.path.join(run_root, f"r{rep}"),
                    max_wall=args.max_wall,
                )
            )

    with open(os.path.join(run_root, "results.json"), "w") as f:
        json.dump(recs, f, indent=1)

    n = len(recs)
    ok = sum(r["passed"] for r in recs)
    hours = sum(r["wall_s"] for r in recs) / 3600 or 1e-9
    print(
        f"\n{n} runs  pass={ok}/{n} ({100 * ok / n:.0f}%)  "
        f"verified tasks/hour={ok / hours:.2f}  "
        f"peak_vram={max(r['peak_vram_mib'] for r in recs)} MiB  "
        f"invalid_actions={sum(r['invalid_actions'] for r in recs)}"
    )
    print(f"results: {run_root}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
