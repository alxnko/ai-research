"""Generic provenance-writing launcher for controller-run attempts.

    python -m experiments.launcher --manifest M --arm A --run-dir DIR <bench args>
    python -m experiments.launcher --manifest M --arm A --print-hashes

The controller starts this for every arm that has no launcher of its own. It
derives everything it records from the approved manifest and from what it
observes, never from values handed to it:

Before any model call (a failure here exits 4 and leaves `launch-refused.json`
in the run directory; the engine has not started):
  1. the bench arguments are exactly those the manifest derives for this
     (arm, seed, task);
  2. the benchmark code on disk is the registered release
     (`source_identity.verify_source`, including the git tag);
  3. the loaded engine modules are those files, not shadowed copies;
  4. the model's manifest, config and weights digests match the manifest;
  5. the arm's prompt, built from the engine's own base prompt plus the
     manifest's `prompt_edit`, hashes to the pinned values.

Then the unchanged engine runs. The recorder wraps the engine's model-call
function and hashes the system message of every request it actually receives.
Only if every request carried exactly the pinned prompt, and at least one was
made, is `provenance.json` written (exit code is then the engine's own, which
is non-zero for a failed task by design). Otherwise `launch-failed.json` is
written instead and the attempt is left for the controller to quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any

from bench import bench as engine
from experiments import controller, source_identity

REFUSED, FAILED = 4, 5


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_prompt(manifest: dict[str, Any]) -> tuple[str, str]:
    """(engine attribute name, the engine's own base prompt) for the actuator."""
    name = (
        "SYSTEM_PROMPT_TOOLS"
        if manifest["settings"]["actuator"] == "native_tools"
        else "SYSTEM_PROMPT_TEXT"
    )
    return name, getattr(engine, name)


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
            self.calls += 1
            self.system_hashes.add(_sha(msgs[0]["content"]))
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


def pre_inference_problems(
    manifest: dict[str, Any], arm: str, bench_args: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Everything that must hold before the model is called."""
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
    problems += _loaded_module_problems()

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


def _loaded_module_problems() -> list[str]:
    """The engine modules in memory must be the files that were verified."""
    problems = []
    for module in ("bench.bench", "bench.classifier", "bench.telemetry"):
        loaded = sys.modules.get(module)
        expected = os.path.join(source_identity.ROOT, *module.split(".")) + ".py"
        path = getattr(loaded, "__file__", None)
        if path is None or os.path.realpath(path) != os.path.realpath(expected):
            problems.append(f"{module} was loaded from {path}, not {expected}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--print-hashes", action="store_true")
    args, bench_args = parser.parse_known_args(argv)

    manifest = controller.load_manifest(args.manifest)
    if args.arm not in manifest["arms"]:
        print(f"unknown arm {args.arm!r}", file=sys.stderr)
        return REFUSED
    if args.print_hashes:
        print(json.dumps(prompt_hashes(manifest, args.arm), indent=2))
        return 0
    if not args.run_dir or manifest.get("frozen"):
        print(
            "a --run-dir and a runnable (non-frozen) manifest are required",
            file=sys.stderr,
        )
        return REFUSED

    problems, facts = pre_inference_problems(manifest, args.arm, bench_args)
    if problems:
        _write(args.run_dir, "launch-refused.json", {"problems": problems})
        print("launch refused:\n  " + "\n  ".join(problems), file=sys.stderr)
        return REFUSED

    return _execute(manifest, args.arm, args.run_dir, bench_args, facts)


def _execute(
    manifest: dict[str, Any],
    arm: str,
    run_dir: str,
    bench_args: list[str],
    facts: dict[str, Any],
) -> int:
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
    sys.argv = [sys.argv[0], "--run-dir", run_dir, *bench_args]
    try:
        rc = engine.main()
    except SystemExit as exc:  # the engine's own early exits
        rc = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # an engine crash: keep the attempt as evidence
        _write(run_dir, "launch-failed.json", {"engine_crash": repr(exc)})
        print(f"launch failed: engine raised {exc!r}", file=sys.stderr)
        return FAILED

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
