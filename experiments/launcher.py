"""Generic provenance-writing launcher for controller-run attempts.

    python -m experiments.launcher --manifest M --manifest-sha256 H --arm A \\
        --run-dir DIR <bench args>
    python -m experiments.launcher --manifest M --arm A --print-hashes

The controller starts this for every attempt; a manifest cannot name another
launcher. Everything it records is derived from the approved manifest and from
what it observes, never from values handed to it.

Order of events (a failure before step 5 exits 4 and leaves only
`launch-refused.json`: the engine has not started, no model was called):

 1. Bytecode isolation: `sys.pycache_prefix` points at a fresh private
    directory (children inherit it via PYTHONPYCACHEPREFIX), so a planted
    `bench/__pycache__/*.pyc` can never be loaded in place of the verified
    source. The engine is not imported yet.
 2. The benchmark code on disk is the registered release
    (`source_identity.verify_source`, including the annotated tag). This runs
    before the engine is imported, then again after (see 3).
 3. The engine and controller are imported; every loaded `bench.*` module must
    be one of the registered source files, and the verification is repeated.
 4. The manifest is the one the controller approved (hash), the bench arguments
    are exactly those it derives, the inference backend matches the manifest
    (the engine's mock model is refused for a real experiment), the model's
    manifest/config/weights digests match, and the arm's prompt (the engine's
    own base prompt plus the manifest's `prompt_edit`) hashes to its pins.
 5. The unchanged engine runs behind a recorder that hashes the system message
    of every request reaching the model-call function (the step loop and the
    empty-reply retry both go through it). Only if every request carried exactly
    the pinned prompt, and at least one was made, is `provenance.json` written
    (the exit code is then the engine's own: non-zero for a failed task by
    design). Otherwise `launch-failed.json` is written and the controller
    quarantines the attempt.

Hash conventions: `prompt_sha256` is the hash of the prompt *template*, before
the engine expands `{max_steps}` (the convention of the frozen EXP-2 wrapper,
whose recorded hashes are of that kind). `system_prompt_sha256` is the hash of
the expanded text actually transmitted, observed at the model call. They are
different quantities and are not comparable with each other.

Known limit: the observation point is the model-call function's arguments. The
HTTP body is built from that same message list; the server's own model store
(as opposed to the local one the fingerprint is read from) is not inspected.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Generator, Mapping
from typing import Any

from experiments import source_identity  # stdlib only; no bench import

REFUSED, FAILED = 4, 5


@contextlib.contextmanager
def isolated_bytecode() -> Generator[None]:
    """Step 1. Compiled modules next to the sources (bench/__pycache__) are never
    consulted while this is active; imports compile into a private directory
    that child processes (the forkserver) inherit. Scoped, so a process that
    merely imports this module keeps its normal bytecode caching."""
    private = tempfile.mkdtemp(prefix="launcher-pycache-")
    saved = (sys.pycache_prefix, os.environ.get("PYTHONPYCACHEPREFIX"))
    sys.pycache_prefix = private
    os.environ["PYTHONPYCACHEPREFIX"] = private
    try:
        yield
    finally:
        sys.pycache_prefix = saved[0]
        if saved[1] is None:
            os.environ.pop("PYTHONPYCACHEPREFIX", None)
        else:
            os.environ["PYTHONPYCACHEPREFIX"] = saved[1]
        shutil.rmtree(private, ignore_errors=True)


def _engine() -> Any:
    return importlib.import_module("bench.bench")


def _controller() -> Any:
    return importlib.import_module("experiments.controller")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def backend() -> str:
    """The inference backend the engine will use (its own mock criterion)."""
    return "mock" if os.environ.get("BENCH_MOCK_MODEL") == "1" else "ollama"


def base_prompt(manifest: dict[str, Any]) -> tuple[str, str]:
    """(engine attribute name, the engine's own base prompt) for the actuator."""
    name = (
        "SYSTEM_PROMPT_TOOLS"
        if manifest["settings"]["actuator"] == "native_tools"
        else "SYSTEM_PROMPT_TEXT"
    )
    return name, getattr(_engine(), name)


def arm_prompt(manifest: dict[str, Any], arm: str) -> tuple[str, str]:
    """(engine attribute name, the arm's prompt template).

    The edit must apply exactly once, so a changed base prompt cannot be edited
    silently into something else."""
    name, template = base_prompt(manifest)
    edit = manifest["arms"][arm].get("prompt_edit")
    if edit:
        old, new = edit["replace"]
        if template.count(old) != 1:
            raise ValueError(
                "prompt_edit does not apply exactly once to the base prompt"
            )
        template = template.replace(old, new, 1)
    return name, template


def prompt_hashes(manifest: dict[str, Any], arm: str) -> dict[str, str]:
    """Template hash (EXP-2 convention) and the hash of what the model receives."""
    _, template = arm_prompt(manifest, arm)
    sent = template.format(max_steps=manifest["settings"]["steps"])
    return {"prompt_sha256": _sha(template), "system_prompt_sha256": _sha(sent)}


class Recorder:
    """Hashes the system message of every request that reaches the model call."""

    def __init__(self) -> None:
        self.calls = 0
        self.system_hashes: set[str] = set()

    def wrap(self, fn):
        def recorded(*args, **kwargs):
            msgs = kwargs["msgs"] if "msgs" in kwargs else args[1]
            first = msgs[0]
            self.calls += 1
            if first.get("role") == "system":
                self.system_hashes.add(_sha(first["content"]))
            else:  # a request without its system message can never match a pin
                self.system_hashes.add(f"not-a-system-message:{first.get('role')}")
            return fn(*args, **kwargs)

        return recorded


def observed_problems(recorder: Recorder, pinned: str) -> list[str]:
    problems = []
    if recorder.calls < 1:
        problems.append("no model call was observed")
    if recorder.system_hashes != {pinned}:
        problems.append(
            "the system prompt sent to the model differs from the pinned one "
            f"(observed {sorted(recorder.system_hashes)})"
        )
    return problems


def _write(run_dir: str, name: str, doc: dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def loaded_module_problems(
    manifest: dict[str, Any], modules: Mapping[str, Any] | None = None
) -> list[str]:
    """Every `bench` module in memory must be a registered, verified source file
    loaded from this repository, and bytecode must not be read from the tree."""
    problems = []
    if sys.pycache_prefix is None:
        problems.append("bytecode cache is not isolated (sys.pycache_prefix unset)")
    registered = {*manifest["benchmark"]["source_files"]}
    root = os.path.realpath(source_identity.ROOT)
    for name, module in (modules if modules is not None else sys.modules).items():
        if name != "bench" and not name.startswith("bench."):
            continue
        path = getattr(module, "__file__", None)
        rel = os.path.relpath(os.path.realpath(path), root) if path else None
        if rel is None or rel.startswith("..") or rel not in registered:
            problems.append(
                f"{name} was loaded from {path}, not a registered source file"
            )
    return problems


def pre_inference_problems(
    manifest: dict[str, Any],
    arm: str,
    bench_args: list[str],
    modules: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Everything that must hold before the model is called."""
    controller, engine = _controller(), _engine()
    problems: list[str] = []
    facts: dict[str, Any] = {}
    parsed = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parsed.add_argument("--seed", type=int)
    parsed.add_argument("--task")
    known, _ = parsed.parse_known_args(bench_args)
    key = (arm, known.seed, known.task)
    if key not in set(controller.attempts(manifest)):
        return [f"{key} is not an attempt of this manifest"], facts
    expected = controller.bench_argv(manifest, known.seed, known.task)
    if bench_args != expected:
        problems.append("bench arguments differ from the manifest's for this attempt")
    facts["benchmark_argv"] = expected
    facts["seed"], facts["task"] = known.seed, known.task

    problems += source_identity.verify_source(manifest)
    problems += loaded_module_problems(manifest, modules)

    wanted = manifest.get("inference", "ollama")
    facts["inference_backend"] = backend()
    if facts["inference_backend"] != wanted:
        problems.append(
            f"inference backend is {facts['inference_backend']!r} but the manifest "
            f"registers {wanted!r} (BENCH_MOCK_MODEL must not leak into real runs)"
        )

    fingerprint = engine.model_fingerprint(manifest["model"]["tag"])
    facts["model_fingerprint"] = {
        k: fingerprint.get(k)
        for k in ("manifest_digest", "config_digest", "weights_digest", "weights_bytes")
    }
    for k in ("manifest_digest", "config_digest", "weights_digest"):
        if fingerprint.get(k) != manifest["model"][k]:
            problems.append(f"model {k} differs from the manifest")

    try:
        facts.update(prompt_hashes(manifest, arm))
    except ValueError as exc:
        problems.append(str(exc))
    else:
        pin = manifest["arms"][arm]
        if facts["prompt_sha256"] != pin["prompt_sha256"]:
            problems.append("the arm's prompt template differs from its pinned hash")
        if facts["system_prompt_sha256"] != pin["system_prompt_sha256"]:
            problems.append("the prompt the model would receive differs from its pin")
    return problems, facts


def _refuse(run_dir: str | None, problems: list[str]) -> int:
    if run_dir:
        _write(run_dir, "launch-refused.json", {"problems": problems})
    print("launch refused:\n  " + "\n  ".join(problems), file=sys.stderr)
    return REFUSED


def main(argv: list[str] | None = None) -> int:
    with isolated_bytecode():  # before the engine is imported
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--print-hashes", action="store_true")
    args, bench_args = parser.parse_known_args(argv)

    # Step 2, before the engine is imported: the files on disk are the release.
    try:
        with open(args.manifest, encoding="utf-8") as f:
            raw = json.load(f)
        early = source_identity.verify_source(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        early = [f"manifest unreadable or malformed ({type(exc).__name__})"]
    if early and not args.print_hashes:
        return _refuse(args.run_dir, early)

    controller = _controller()
    manifest = controller.load_manifest(args.manifest)
    if args.arm not in manifest["arms"]:
        print(f"unknown arm {args.arm!r}", file=sys.stderr)
        return REFUSED
    if args.print_hashes:
        print(json.dumps(prompt_hashes(manifest, args.arm), indent=2))
        return 0
    gate = [
        text
        for failed, text in (
            (not args.run_dir, "a --run-dir is required"),
            (
                bool(manifest.get("frozen")),
                "a runnable (non-frozen) manifest is required",
            ),
            (
                args.manifest_sha256 != controller.manifest_sha256(manifest),
                "the manifest is not the one the controller approved",
            ),
        )
        if failed
    ]
    problems, facts = (
        ([], {}) if gate else pre_inference_problems(manifest, args.arm, bench_args)
    )
    if gate or problems:
        return _refuse(args.run_dir, [*gate, *problems])
    return _execute(manifest, args.arm, args.run_dir, bench_args, facts)


def _execute(
    manifest: dict[str, Any],
    arm: str,
    run_dir: str,
    bench_args: list[str],
    facts: dict[str, Any],
) -> int:
    controller, engine = _controller(), _engine()
    attempt = controller.attempt_id(manifest, arm, facts["seed"], facts["task"])
    _write(
        run_dir,
        "launch-started.json",
        {"attempt_id": attempt, "started": time.time(), **facts},
    )

    prompt_attr, template = arm_prompt(manifest, arm)
    recorder = Recorder()
    setattr(engine, prompt_attr, template)
    engine.chat_stream = recorder.wrap(engine.chat_stream)
    engine._mock_chat_stream = recorder.wrap(engine._mock_chat_stream)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--run-dir", run_dir, *bench_args]
    try:
        rc = engine.main()
    except SystemExit as exc:  # the engine's own early exits
        rc = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # an engine crash: keep the attempt as evidence
        _write(run_dir, "launch-failed.json", {"engine_crash": repr(exc)})
        print(f"launch failed: engine raised {exc!r}", file=sys.stderr)
        return FAILED
    finally:
        sys.argv = saved_argv

    failures = observed_problems(recorder, facts["system_prompt_sha256"])
    if failures:
        _write(run_dir, "launch-failed.json", {"problems": failures})
        print("launch failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return FAILED
    bench = manifest["benchmark"]
    _write(
        run_dir,
        manifest.get("provenance_file", "provenance.json"),
        {
            "experiment": manifest["experiment"],
            "policy": arm,
            "attempt_id": attempt,
            "manifest_sha256": controller.manifest_sha256(manifest),
            "benchmark_tag": bench["tag"],
            "benchmark_core_commit": bench["commit"],
            "source_sha256": bench["source_sha256"],
            "source_verification": "git+filesystem",
            "inference_backend": facts["inference_backend"],
            "model_fingerprint": facts["model_fingerprint"],
            "prompt_sha256": facts["prompt_sha256"],
            "system_prompt_sha256": next(iter(recorder.system_hashes)),
            "model_calls": recorder.calls,
            "benchmark_argv": facts["benchmark_argv"],
        },
    )
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
