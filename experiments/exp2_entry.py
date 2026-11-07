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

    # Fail closed on accidental configuration drift between paired arms.
    checks = argparse.ArgumentParser(add_help=False)
    checks.add_argument("--model")
    checks.add_argument("--actuator")
    checks.add_argument("--task")
    checks.add_argument("--ctx", type=int)
    checks.add_argument("--ngpu", type=int)
    checks.add_argument("--steps", type=int)
    checks.add_argument("--max-wall", type=int)
    checks.add_argument("--strict-isolation", action="store_true")
    checks.add_argument("--temperature", type=float)
    checks.add_argument("--top-p", type=float)
    checks.add_argument("--top-k", type=int)
    checks.add_argument("--repeat-penalty", type=float)
    checks.add_argument("--seed", type=int)
    checks.add_argument("--sampling-profile")
    parsed, unknown = checks.parse_known_args(bench_args)
    expected = {
        "model": "qwen3.5:4b-q4_K_M",
        "actuator": "native_tools",
        "ctx": 16384,
        "ngpu": 99,
        "steps": 20,
        "max_wall": 600,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "sampling_profile": "exp2-coding-v1",
        "strict_isolation": True,
    }
    for key, value in expected.items():
        if getattr(parsed, key) != value:
            raise SystemExit(f"EXP-2 configuration mismatch: {key} must be {value!r}")
    allowed_tasks = {
        "backend-auth-token", "backend-idempotency",
        "debugging-off-by-one", "debugging-money-round",
        "refactor-parser", "refactor-validators",
    }
    if parsed.task not in allowed_tasks or parsed.seed not in {42, 43, 44}:
        raise SystemExit("EXP-2 requires one preregistered task and seed 42/43/44")
    if unknown:
        raise SystemExit(f"Unregistered benchmark arguments: {unknown!r}")

    expected_dir = os.path.join(
        benchmark.ROOT, "runs", "exp2", args.exp2_policy,
        f"seed{parsed.seed}", parsed.task,
    )
    if os.path.abspath(args.run_dir) != expected_dir:
        raise SystemExit(f"EXP-2 output must be {expected_dir}")

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
