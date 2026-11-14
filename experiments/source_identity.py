"""Code identity of the benchmark that is about to execute (run-time only).

Contract, checked before any model inference. A manifest registers:

    benchmark.tag           annotated tag of the benchmark release
    benchmark.commit        the commit that tag must resolve to (40 hex)
    benchmark.source_files  executable benchmark files (engine, scoring,
                            sandbox and supporting modules)
    benchmark.source_sha256 digest over those files plus the suite manifest

`verify_source` then fails closed unless, for the files actually on disk:

 1. the digest of `source_files` + the suite manifest equals `source_sha256`;
 2. every selected task directory equals the digest the suite pins for it
    (stray files such as __pycache__ change a directory digest);
 3. no source file is a symlink or missing;
 4. the tag exists and resolves to the registered commit; and
 5. each of those files, and each selected task, is byte-identical to the
    object stored under that commit (filesystem content versus the release,
    not "git says the tree is clean").

The checkout's HEAD is irrelevant: newer controller code may execute a frozen
benchmark as long as the benchmark files themselves are identical. Offline
commands (`check`, `plan`, `analyze`) never call this.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dir_digest(path: str) -> str:
    """Same algorithm as bench.task_digest, for any directory."""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).encode())
            with open(full, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def source_paths(manifest: dict[str, Any]) -> list[str]:
    bench = manifest["benchmark"]
    return sorted(
        {*bench.get("source_files", []), f"bench/suites/{bench['suite']}.json"}
    )


def files_digest(root: str, paths: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(paths):
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _git(root: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, check=False, timeout=60
    )


def resolve_tag(root: str, tag: str) -> str | None:
    proc = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def _tag_task_digest(root: str, commit: str, task: str) -> str | None:
    proc = _git(root, "archive", commit, f"bench/tasks/{task}")
    if proc.returncode != 0:
        return None
    with tempfile.TemporaryDirectory(prefix="tag-task-", dir=_scratch()) as tmp:
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        return dir_digest(os.path.join(tmp, "bench", "tasks", task))


def _scratch() -> str | None:
    return "/var/tmp" if os.path.isdir("/var/tmp") else None


def _suite_pins(root: str, suite: str) -> dict[str, str]:
    with open(
        os.path.join(root, "bench", "suites", f"{suite}.json"), encoding="utf-8"
    ) as f:
        return json.load(f)["tasks"]


def verify_source(
    manifest: dict[str, Any], root: str = ROOT, *, use_git: bool = True
) -> list[str]:
    """Why the benchmark on disk is not the registered one (empty: identical)."""
    bench = manifest["benchmark"]
    if not (bench.get("source_files") and bench.get("source_sha256")):
        return [
            "manifest registers no source identity (frozen manifests are analysis-only)"
        ]
    problems: list[str] = []
    paths = source_paths(manifest)
    for rel in paths:
        full = os.path.join(root, rel)
        if os.path.islink(full):
            problems.append(f"{rel} is a symlink")
        elif not os.path.isfile(full):
            problems.append(f"{rel} is missing")
    if problems:
        return problems
    if files_digest(root, paths) != bench["source_sha256"]:
        problems.append("benchmark source files differ from the registered digest")
    try:
        pins = _suite_pins(root, bench["suite"])
    except (OSError, ValueError, KeyError) as exc:
        return [*problems, f"suite manifest unreadable ({type(exc).__name__})"]
    for task in manifest["tasks"]:
        tdir = os.path.join(root, "bench", "tasks", task)
        if task not in pins:
            problems.append(f"task {task!r} is not in the suite")
        elif not os.path.isdir(tdir) or os.path.islink(tdir):
            problems.append(f"task {task!r} is missing or a symlink")
        elif dir_digest(tdir) != pins[task]:
            problems.append(f"task {task!r} fixtures differ from the suite pin")
    if use_git:
        problems += _verify_against_tag(manifest, root, paths, pins)
    return problems


def _verify_against_tag(
    manifest: dict[str, Any], root: str, paths: list[str], pins: dict[str, str]
) -> list[str]:
    bench = manifest["benchmark"]
    commit = resolve_tag(root, bench["tag"])
    if commit is None:
        return [f"tag {bench['tag']!r} not found (fetch tags: git fetch --tags)"]
    if commit != bench["commit"]:
        return [
            f"tag {bench['tag']!r} resolves to {commit}, registered {bench['commit']}"
        ]
    problems = []
    for rel in paths:
        proc = _git(root, "cat-file", "blob", f"{commit}:{rel}")
        if proc.returncode != 0:
            problems.append(f"{rel} does not exist at the tag")
            continue
        with open(os.path.join(root, rel), "rb") as f:
            if f.read() != proc.stdout:
                problems.append(f"{rel} differs from the tag")
    for task in manifest["tasks"]:
        if task in pins and _tag_task_digest(root, commit, task) != pins[task]:
            problems.append(f"task {task!r} at the tag differs from the suite pin")
    return problems


def registration(
    tag: str, suite: str, files: list[str], root: str = ROOT
) -> dict[str, Any]:
    """The values to register in a manifest, computed from the *tag's* objects
    (not the working tree): benchmark.commit and benchmark.source_sha256."""
    commit = resolve_tag(root, tag)
    if commit is None:
        raise ValueError(f"tag {tag!r} not found")
    paths = sorted({*files, f"bench/suites/{suite}.json"})
    h = hashlib.sha256()
    for rel in paths:
        proc = _git(root, "cat-file", "blob", f"{commit}:{rel}")
        if proc.returncode != 0:
            raise ValueError(f"{rel} does not exist at {tag}")
        h.update(rel.encode())
        h.update(proc.stdout)
    return {
        "commit": commit,
        "source_files": sorted(files),
        "source_sha256": h.hexdigest(),
    }


if __name__ == "__main__":
    import sys

    tag, suite, *files = sys.argv[1:]
    print(json.dumps(registration(tag, suite, files), indent=2))
