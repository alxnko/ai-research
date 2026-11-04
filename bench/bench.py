#!/usr/bin/env python3
"""Run benchmark tasks against local LLMs through Ollama, inside a bubblewrap sandbox.

Usage:
    python3 bench.py --model qwen2.5-coder:7b --ctx 32768 --ngpu 99 --steps 20
    python3 bench.py --task backend-rate-limit --steps 10
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
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(ROOT, "tasks")
RUNS = os.path.join(ROOT, "runs")

# Discover python home dynamically from sys.executable
PYTHON_HOME = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))

SYSTEM_PROMPT = """You are an expert software engineer.
You solve tasks in a Linux repository interacting exclusively through shell commands.

To run a command:
RUN: <command>

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
    "/etc/ssl",
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
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--",
        *cmd_args,
    ]
    return args


def run_sandboxed(workdir, cmd, timeout, outdir=None):
    """Run under bwrap. The command is written to an isolated runner script
    to preserve heredocs, multiline syntax, quotes, and exit codes without
    any shell concatenation. The exit code is determined authoritatively
    by the parent harness via bwrap process status."""
    cleanup_outdir = False
    if not outdir:
        outdir = tempfile.mkdtemp(prefix="bench-out-")
        cleanup_outdir = True
    os.makedirs(outdir, exist_ok=True)
    stdout_path = os.path.join(outdir, "stdout")
    script_path = os.path.join(outdir, "run.sh")
    for path in (stdout_path, script_path):
        if os.path.exists(path):
            os.remove(path)

    normalized_cmd = cmd.rstrip("\r\n") + "\n"
    script_content = (
        "#!/usr/bin/env bash\n"
        "exec > /mnt/out/stdout 2>&1\n"
        "ulimit -t 120 2>/dev/null || true\n"
        "ulimit -f 16384 2>/dev/null || true\n"
        "ulimit -u 512 2>/dev/null || true\n"
        "ulimit -v 4194304 2>/dev/null || true\n"
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
                "PATH": "/usr/bin:/bin:/opt/python/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
        code = proc.returncode
    except subprocess.TimeoutExpired:
        if cleanup_outdir:
            shutil.rmtree(outdir, ignore_errors=True)
        return 124, f"<timeout after {timeout}s>"

    text = ""
    try:
        if os.path.exists(stdout_path):
            size = os.path.getsize(stdout_path)
            with open(stdout_path, "rb") as f:
                if size > OUTPUT_CAP:
                    f.seek(size - OUTPUT_CAP)
                    text = f"<output truncated, last {OUTPUT_CAP} bytes>\n"
                text += f.read().decode("utf-8", "replace")
    except OSError:
        pass

    if not text and code != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        if err:
            text = f"<bwrap error ({code}): {err}>"

    if cleanup_outdir:
        shutil.rmtree(outdir, ignore_errors=True)
    return code, text


# ---------- vram monitoring ----------


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
    meta["id"] = task_id
    meta["dir"] = path
    return meta


def all_tasks():
    return sorted(
        t
        for t in os.listdir(TASKS)
        if os.path.isfile(os.path.join(TASKS, t, "meta.json"))
    )


SKIP_DIRS = {".git", "_hidden"}


def _classify_entry(path, root):
    """Classify a filesystem entry with its permissions and digest."""
    rel = os.path.relpath(path, root)
    try:
        st = os.lstat(path)
    except OSError:
        return rel, None
    mode = st.st_mode
    perm = stat.S_IMODE(mode)
    if stat.S_ISLNK(mode):
        try:
            target = os.readlink(path)
        except OSError:
            target = ""
        info = {"type": "symlink", "target": target, "perm": perm}
    elif stat.S_ISDIR(mode):
        info = {"type": "dir", "perm": perm}
    elif stat.S_ISREG(mode):
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            sha = digest.hexdigest()
        except OSError:
            sha = ""
        info = {"type": "file", "sha256": sha, "perm": perm}
    elif stat.S_ISFIFO(mode):
        info = {"type": "fifo", "perm": perm}
    elif stat.S_ISSOCK(mode):
        info = {"type": "socket", "perm": perm}
    else:
        info = {"type": "special", "perm": perm}
    return rel, info


def snapshot(root):
    """Deep structural inventory of the workspace.

    Classifies files, symlinks (including symlinked directories), fifos,
    sockets, special files, and directory permissions to track agent modifications.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept_dirs = []
        for d in dirnames:
            if d in SKIP_DIRS:
                continue
            rel, info = _classify_entry(os.path.join(dirpath, d), root)
            if info:
                out[rel] = info
                if info["type"] == "dir":
                    kept_dirs.append(d)
        dirnames[:] = sorted(kept_dirs)

        for name in filenames:
            rel, info = _classify_entry(os.path.join(dirpath, name), root)
            if info:
                out[rel] = info
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
    work = os.path.join(run_dir, "work")
    shutil.copytree(os.path.join(meta["dir"], "repo"), work)
    with open(os.path.join(run_dir, "baseline.json"), "w") as f:
        json.dump(snapshot(work), f, indent=2)
    return work


def install_tests(meta, target):
    """Mount the hidden tests at _hidden/ in the target workspace."""
    hidden = os.path.join(target, "_hidden")
    if os.path.islink(hidden):
        raise RuntimeError("workspace contains symlinked _hidden")
    if os.path.exists(hidden):
        raise RuntimeError(f"workspace already contains {hidden}")
    tests_src = os.path.join(meta["dir"], "tests")
    if not os.path.isdir(tests_src):
        tests_src = os.path.join(meta["dir"], "hidden")
    shutil.copytree(tests_src, hidden)
    open(os.path.join(hidden, "__init__.py"), "a").close()


def score(meta, work, run_dir):
    """Clean copy of candidate files + clean install of tests + sandbox run."""
    scoring_dir = os.path.join(run_dir, "score")
    clean_copy(work, scoring_dir)
    install_tests(meta, scoring_dir)
    test_cmd = meta.get("test_cmd", "python3 -m unittest discover _hidden")
    code, out = run_sandboxed(scoring_dir, test_cmd, timeout=60)
    passed = code == 0
    return passed, f"exit={code}\n{out}"


def policy_check(meta, work, run_dir):
    """Compare the workspace against the baseline taken before the agent ran.

    Detects unauthorized files, new symlinks (including directory symlinks),
    FIFOs, sockets, and unauthorized permission changes.
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
            if (
                (allowed and k not in allowed)
                or curr["type"] not in ("file", "dir")
                or perm_changed
            ):
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
    }


# ---------- agent loop ----------


def run_task(
    meta,
    model,
    ctx,
    ngpu,
    max_steps,
    max_wall,
    run_dir,
    verbose=True,
    chat_fn=None,
):
    """One run of one task. Everything needed to replicate lives in run_dir."""
    os.makedirs(run_dir, exist_ok=True)
    work = prepare(meta, run_dir)
    prompt_text = meta.get("prompt") or meta.get("instruction", "")
    user = f"{prompt_text}\n\nWork in /mnt/work. The repo is already checked out."

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_steps=max_steps)},
        {"role": "user", "content": user},
    ]
    transcript = [{"role": "system", "content": msgs[0]["content"]}]
    transcript.append({"role": "user", "content": user})

    if chat_fn is None:
        chat_fn = chat

    sampler = VramSampler()
    sampler.start()

    t0 = time.time()
    steps = 0
    tokens = 0
    tool_calls = 0
    invalid = 0
    timed_out = False

    try:
        while steps < max_steps:
            rem = max_wall - (time.time() - t0)
            if rem <= 0:
                timed_out = True
                break
            reply, thinking, ntok = chat_fn(
                model, msgs, ctx, ngpu, timeout=min(1800, int(rem))
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
                        "content": (
                            "You did not output an action. Emit exactly one command "
                            "starting with 'RUN: ' or the single word 'DONE'."
                        ),
                    },
                ]
                reply, _, ntok2 = chat_fn(
                    model, retry_msgs, ctx, ngpu, timeout=min(1800, rem_retry)
                )
                tokens += ntok2
                kind, cmd = parse_action(reply)
            transcript.append(
                {"step": steps, "reply": reply, "thinking": thinking, "action": kind}
            )

            if kind == "done":
                break
            if kind != "run" or not cmd:
                invalid += 1
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
            feed = f"[exit={code}]\n{out}" if out else f"[exit={code}]"
            transcript[-1]["code"] = code
            transcript[-1]["output"] = out
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": feed})
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
    with open(os.path.join(run_dir, "test_output.txt"), "w") as f:
        f.write(test_out)

    rec = {
        "task": meta["id"],
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
        "timed_out": timed_out,
        "changed_files": diff["changed"],
        "added_files": diff["added"],
        "removed_files": diff["removed"],
        "forbidden_files": diff["forbidden"],
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
    """Provenance: safe allowlist of environment details; sensitive data redacted."""
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
    args = p.parse_args()

    ids = [args.task] if args.task else all_tasks()
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

    results_path = os.path.join(run_dir, "results.json")
    recs = []
    completed_keys = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                recs = json.load(f)
                completed_keys = {(r.get("repeat", 0), r.get("task")) for r in recs}
                print(f"Resuming run: found {len(recs)} already completed task(s).")
        except Exception:
            recs = []

    for rep in range(args.repeat):
        for tid in ids:
            if (rep, tid) in completed_keys:
                continue
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
            rec["task"] = tid
            rec["repeat"] = rep
            recs.append(rec)
            # Atomic update of results.json after each task
            tmp_results = results_path + f".tmp.{os.getpid()}"
            with open(tmp_results, "w") as f:
                json.dump(recs, f, indent=2)
            os.replace(tmp_results, results_path)

    ok = sum(1 for r in recs if r["passed"])
    n = len(recs)
    total_h = sum(r["total_wall_s"] for r in recs) / 3600 or 1e-9
    agent_h = sum(r["agent_wall_s"] for r in recs) / 3600 or 1e-9
    verify_h = sum(r["verification_wall_s"] for r in recs) / 3600 or 1e-9
    print(
        f"\n{n} runs  pass={ok}/{n} ({100 * ok / n:.0f}%)  "
        f"verified tasks/hour={ok / total_h:.2f}  "
        f"(agent: {agent_h * 3600:.1f}s, verify: {verify_h * 3600:.1f}s)  "
        f"peak_vram={max((r['peak_vram_mib'] for r in recs), default=0)} MiB  "
        f"invalid_actions={sum(r['invalid_actions'] for r in recs)}"
    )
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
