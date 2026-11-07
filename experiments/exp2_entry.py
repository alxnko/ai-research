"""EXP-2 prompt-only arm selector, keeping benchmark v0.3 core unchanged.

Use as: uv run python -m experiments.exp2_entry --exp2-policy control|verification \
    --run-dir runs/exp2/<arm>/seed42/<task> [standard bench.py flags]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from bench import bench as benchmark


BASE_PROMPT = benchmark.SYSTEM_PROMPT_TOOLS
OLD_SENTENCE = "Ensure all changes are verified before finishing."
NEW_SENTENCE = (
    "Before finishing, verify the requested behavior using available public "
    "tests and targeted checks for relevant boundary conditions, input types, "
    "error handling, and behavioral invariants. If a check fails, inspect the "
    "failure and repair the implementation. Do not access hidden evaluation tests."
)

if BASE_PROMPT.count(OLD_SENTENCE) != 1:
    raise RuntimeError("Frozen v0.3 baseline prompt has changed unexpectedly")

VERIFY_PROMPT = BASE_PROMPT.replace(OLD_SENTENCE, NEW_SENTENCE, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EXP-2 control/verification policy wrapper (same bench core)",
        add_help=False,
    )
    parser.add_argument("--exp2-policy", choices=("control", "verification"), required=True)
    parser.add_argument("--run-dir", required=True)
    args, bench_args = parser.parse_known_args()

    if not any(x == "--actuator" or x.startswith("--actuator=") for x in bench_args):
        raise SystemExit("EXP-2 requires --actuator native_tools")
    if not any(x == "--task" or x.startswith("--task=") for x in bench_args):
        raise SystemExit("EXP-2 requires one explicit --task per run")
    if not any(x == "--strict-isolation" for x in bench_args):
        raise SystemExit("EXP-2 requires --strict-isolation")

    prompt = BASE_PROMPT if args.exp2_policy == "control" else VERIFY_PROMPT
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    resolved = os.path.abspath(args.run_dir)
    os.makedirs(resolved, exist_ok=True)

    # Refuse silent arm/prompt changes or altered invocations on resume.
    record_path = os.path.join(resolved, "exp2_policy.json")
    record = {
        "experiment": "EXP-2",
        "benchmark_tag": "v0.3",
        "benchmark_core_commit": "dd3f6bcde28e6ba8db8b8c83d2d20f2f91562e00",
        "policy": args.exp2_policy,
        "prompt_sha256": digest,
        "benchmark_argv": bench_args,
    }
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as f:
            old_record = json.load(f)
        if old_record != record:
            raise SystemExit("Existing EXP-2 output has a different policy or CLI invocation")
    else:
        with open(record_path, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)

    benchmark.SYSTEM_PROMPT_TOOLS = prompt
    print(f"EXP-2 policy={args.exp2_policy} prompt_sha256={digest}", flush=True)
    sys.argv = [sys.argv[0], "--run-dir", args.run_dir, *bench_args]
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
