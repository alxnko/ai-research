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
import stat
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
    os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))),
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
- Exactly one RUN or DONE per reply.
- Never edit the test runner or files you are not asked to modify.
- Do not run interactive commands (vi, nano, less, python without -c).
- Make sure all your changes are saved to disk before saying DONE.
- You have at most {max_steps} steps. Be concise.
"""


def chat(model, msgs, ctx, ngpu, timeout=1800):
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {
            "num_ctx": ctx,
            "num_gpu": ngpu,
            "temperature": 0.0,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    msg = data.get("message", {})
    return (
        msg.get("content", ""),
        msg.get("thinking", ""),
        data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    )


def parse_action(reply):
    """Extract the agent action from the model output.

    Handles explicit format:
        RUN: <command>
        DONE

    Also handles common minor formatting variations:
        - fenced bash code blocks
        - XML tags <command>...</command> or <result>...</result>
        - DONE with trailing commentary
    """
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


# Host files needed inside the sandbox for standard library name resolution,
# timezone data, and dynamic linker operation. Read-only.
ETC_FILES = [
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/ssl/certs",
]

OUTPUT_CAP = 4096  # bytes of command output shown to the model


def bwrap_cmd(workdir, cmd_args, outdir=None):
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
        *cmd_args,
    ]
    return args


def run_sandboxed(workdir, cmd, timeout, outdir=None):
    """Run under bwrap. The command is written to an isolated runner script
    to preserve heredocs, multiline syntax, quotes, and exit codes without
    any shell concatenation."""
    cleanup_outdir = False
    if not outdir:
        outdir = tempfile.mkdtemp(prefix="bench-out-")
        cleanup_outdir = True
    os.makedirs(outdir, exist_ok=True)
    stdout_path = os.path.join(outdir, "stdout")
    rc_path = os.path.join(outdir, "rc")
    script_path = os.path.join(outdir, "run.sh")
    for path in (stdout_path, rc_path, script_path):
        if os.path.exists(path):
            os.remove(path)

    normalized_cmd = cmd.rstrip("\r\n") + "\n"
    script_content = (
        "#!/usr/bin/env bash\n"
        "exec > /mnt/out/stdout 2>&1\n"
        "ulimit -t 120 2>/dev/null || true\n"
        "ulimit -f 16384 2>/dev/null || true\n"
        "ulimit -u 512 2>/dev/null || true\n"
        "trap 'echo $? > /mnt/out/rc' EXIT\n"
        f"{normalized_cmd}"
    )
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    try:
        proc = subprocess.run(
            bwrap_cmd(workdir, ["bash", "/mnt/out/run.sh"], outdir),
            capture_output=True,
            timeout=timeout,
            env={
                "PATH": "/opt/python/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
            },
        )
    except subprocess.TimeoutExpired:
        if cleanup_outdir:
            shutil.rmtree(outdir, ignore_errors=True)
        return 124, f"<timeout after {timeout}s>"

    code = 124
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as f:
                code = int(f.read().strip())
        except OSError, ValueError:
            code = 124
    else:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        msg = (
            f"<bwrap failed ({proc.returncode}): {err}>"
            if err
            else "<killed before reporting an exit code>"
        )
        if cleanup_outdir:
            shutil.rmtree(outdir, ignore_errors=True)
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

    if cleanup_outdir:
        shutil.rmtree(outdir, ignore_errors=True)
    return code, text


# ---------- vram sampler ----------


class VramSampler(threading.Thread):
    """Background poller that tracks peak GPU memory during the run."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            v = self._read()
            self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def _read(self):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0


# ---------- tasks & workspaces ----------


def load_task(task_id):
    path = os.path.join(TASKS, task_id)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    meta["dir"] = path
    return meta


def all_tasks():
    return sorted(
        t
        for t in os.listdir(TASKS)
        if os.path.isfile(os.path.join(TASKS, t, "meta.json"))
    )


SKIP_DIRS = {".git", "_hidden"}


def snapshot(root):
    """Deep structural inventory of the workspace.

    Classifies files, symlinks, fifos, sockets, and special files with
    cryptographic hashes and permissions to track agent modifications.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            perm = stat.S_IMODE(st.st_mode)
            if stat.S_ISLNK(st.st_mode):
                try:
                    target = os.readlink(path)
                except OSError:
                    target = ""
                out[rel] = {"type": "symlink", "target": target, "perm": perm}
            elif stat.S_ISREG(st.st_mode):
                digest = hashlib.sha256()
                try:
                    with open(path, "rb") as f:
                        for block in iter(lambda: f.read(1 << 20), b""):
                            digest.update(block)
                    out[rel] = {
                        "type": "file",
                        "sha256": digest.hexdigest(),
                        "perm": perm,
                    }
                except OSError:
                    out[rel] = {"type": "file", "sha256": "", "perm": perm}
            elif stat.S_ISFIFO(st.st_mode):
                out[rel] = {"type": "fifo", "perm": perm}
            elif stat.S_ISSOCK(st.st_mode):
                out[rel] = {"type": "socket", "perm": perm}
            else:
                out[rel] = {"type": "special", "perm": perm}
    return out


def clean_copy(src, dst):
    """Rebuild a workspace from regular files only, so symlinks and special
    files created by the agent can never reach the scorer."""
    os.makedirs(dst, exist_ok=True)
    snap = snapshot(src)
    for rel, info in snap.items():
        if info.get("type") != "file":
            continue
        src_path = os.path.join(src, rel)
        try:
            st = os.lstat(src_path)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
        except OSError:
            continue
        target = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(target) or dst, exist_ok=True)
        shutil.copy2(src_path, target)


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

    Detects unauthorized files, new symlinks, fifos, sockets, and modifications.
    """
    with open(os.path.join(run_dir, "baseline.json")) as f:
        baseline = json.load(f)
    current = snapshot(work)

    changed = []
    forbidden = []
    allowed = set(meta.get("allowed_files", []))

    for k, curr in current.items():
        base = baseline.get(k)
        if base is None:
            if curr["type"] != "file" or (allowed and k not in allowed):
                forbidden.append(k)
        elif curr != base:
            changed.append(k)
            perm_changed = curr.get("perm") != base.get("perm")
            if (allowed and k not in allowed) or curr["type"] != "file" or perm_changed:
                forbidden.append(k)

    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    for k in removed:
        if allowed and k not in allowed:
            forbidden.append(k)

    changed = sorted(set(changed))
    forbidden = sorted(set(forbidden))

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "forbidden": forbidden,
        "policy_enforced": bool(allowed),
    }


# ---------- agent loop ----------


def run_task(meta, model, ctx, ngpu, max_steps, max_wall, run_dir, verbose=True):
    """One run of one task. Everything needed to replicate lives in run_dir."""
    os.makedirs(run_dir, exist_ok=True)
    work = prepare(meta, run_dir)
    user = (
        f"Task: {meta['instruction']}\n\n"
        "Work in /mnt/work. The repo is already checked out."
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
                rem_retry = max(1, int(max_wall - (time.time() - t0)))
                retry_msgs = [
                    *msgs,
                    {"role": "assistant", "content": thinking},
                    {
                        "role": "user",
                        "content": "Now write the action. Reply with exactly one line "
                        "starting with 'RUN: ' or the single word 'DONE'.",
                    },
                ]
                reply, _, ntok2 = chat(
                    model, retry_msgs, ctx, ngpu, timeout=min(1800, rem_retry)
                )
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
                min(
                    meta.get("cmd_timeout_s", 60),
                    int(max_wall - (time.time() - t0)),
                ),
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

    t_agent_end = time.time()
    agent_wall_s = t_agent_end - t0
    t_verify_start = time.time()
    tests_ok, test_out = score(meta, work, run_dir)
    diff = policy_check(meta, work, run_dir)
    verification_wall_s = time.time() - t_verify_start
    total_wall_s = time.time() - t0
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
        "agent_wall_s": round(agent_wall_s, 2),
        "verification_wall_s": round(verification_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
        "wall_s": round(total_wall_s, 2),
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
        timing = (
            f"{rec['wall_s']:6.1f}s "
            f"(agent: {agent_wall_s:5.1f}s, verify: {verification_wall_s:4.1f}s)"
        )
        print(
            f"  {'PASS' if passed else 'FAIL'}  {meta['id']:24s} "
            f"{timing} "
            f"{rec['steps']:2d} steps {rec['tokens']:6d} tok "
            f"{'REGRESSION' if diff['forbidden'] else ''}"
            f"{' TIMEOUT' if timed_out else ''}"
            f"{'' if tests_ok or diff['forbidden'] else ' tests-failed'}",
            flush=True,
        )
    return rec


# ---------- environment & provenance ----------


def _sh(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
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


def model_fingerprint(model):
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


def model_digest(model):
    """Convenience wrapper returning weights digest or config digest."""
    fp = model_fingerprint(model)
    return fp["weights_digest"] or fp["config_digest"]


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
    tag = f"-{args.tag}" if args.tag else ""
    run_dir = os.path.join(RUNS, f"{stamp}{tag}")
    os.makedirs(run_dir, exist_ok=True)

    env = environment()
    fp = model_fingerprint(args.model)
    env["model_fingerprint"] = fp
    env["model_digest"] = fp["weights_digest"] or fp["config_digest"]

    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)

    print(
        f"RUN {run_dir}\n"
        f"model={args.model} ({env['model_digest']})  "
        f"ctx={args.ctx}  ngpu={args.ngpu}  tasks={len(ids)}  repeat={args.repeat}\n"
        f"host={env['gpu']} (driver {env['nvidia_driver']})  "
        f"python={env['python']}\n",
        flush=True,
    )

    recs = []
    for rep in range(args.repeat):
        for tid in ids:
            meta = load_task(tid)
            tdir = os.path.join(run_dir, f"r{rep}", tid)
            rec = run_task(
                meta,
                args.model,
                args.ctx,
                args.ngpu,
                args.steps,
                args.max_wall,
                tdir,
            )
            rec["repeat"] = rep
            recs.append(rec)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(recs, f, indent=2)

    ok = sum(1 for r in recs if r["passed"])
    n = len(recs)
    total_h = sum(r["total_wall_s"] for r in recs) / 3600 or 1e-9
    agent_h = sum(r["agent_wall_s"] for r in recs) / 3600 or 1e-9
    verify_h = sum(r["verification_wall_s"] for r in recs) / 3600 or 1e-9
    print(
        f"\n{n} runs  pass={ok}/{n} ({100 * ok / n:.0f}%)  "
        f"verified tasks/hour={ok / total_h:.2f}  "
        f"(agent: {agent_h * 3600:.1f}s, verify: {verify_h * 3600:.1f}s)  "
        f"peak_vram={max(r['peak_vram_mib'] for r in recs)} MiB  "
        f"invalid_actions={sum(r['invalid_actions'] for r in recs)}"
    )
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
