"""Two-axis task run failure classification separating outcomes from causes."""

from __future__ import annotations

from typing import Any

VALID_OUTCOMES = {
    "passed",
    "hidden_test_failure",
    "invalid_action",
    "timeout",
    "repeated_command",
    "premature_completion",
    "forbidden_file_change",
    "inference_failure",
    "memory_failure",
}

VALID_CAUSES = {
    "insufficient_reasoning",
    "quantization_degradation",
    "context_loss",
    "tool_format_incompatibility",
    "prompt_mismatch",
    "test_fixture_defect",
    "resource_constraint",
    "unknown",
}


def _detect_repeated_command_loop(
    steps: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Detect if agent repeated the exact same command 3+ times with same output."""
    if len(steps) < 3:
        return False, ""
    recent = steps[-3:]
    cmds = [s.get("command") for s in recent]
    outputs = [s.get("output") for s in recent]
    match_cmd = cmds[0] and cmds[0] == cmds[1] == cmds[2]
    match_out = outputs[0] == outputs[1] == outputs[2]
    if match_cmd and match_out:
        return True, f"command '{cmds[0]}' executed 3 times with identical output"
    return False, ""


def classify_task_run(
    passed: bool,
    timed_out: bool,
    step_count: int,
    max_steps: int,
    invalid_actions: int,
    forbidden_files: list[str],
    changed_files: list[str],
    exit_status: str,
    steps: list[dict[str, Any]],
    num_ctx: int = 32768,
    telemetry_records: list[dict[str, Any]] | None = None,
    inference_error: str | None = None,
) -> dict[str, Any]:
    """Classify a completed task trajectory into outcome and evidence-backed cause."""
    evidence: list[str] = []

    # 1. Determine Observed Outcome
    if inference_error:
        outcome = "inference_failure"
        evidence.append(f"inference transport error: {inference_error}")
    elif passed:
        outcome = "passed"
    elif "cgroup: oom-kill" in exit_status.lower() or "killed" in exit_status.lower():
        outcome = "memory_failure"
        evidence.append(f"command killed by memory limit: {exit_status}")
    elif forbidden_files:
        outcome = "forbidden_file_change"
        evidence.append(f"modified forbidden files: {', '.join(forbidden_files)}")
    elif timed_out:
        outcome = "timeout"
        evidence.append("exceeded wall-clock time budget")
    else:
        is_loop, loop_desc = _detect_repeated_command_loop(steps)
        if is_loop:
            outcome = "repeated_command"
            evidence.append(loop_desc)
        elif invalid_actions > 0 and (step_count >= max_steps or invalid_actions >= 3):
            outcome = "invalid_action"
            evidence.append(f"agent aborted after {invalid_actions} invalid action(s)")
        elif not changed_files and (
            step_count <= 2 or not any(s.get("command") for s in steps)
        ):
            outcome = "premature_completion"
            evidence.append("agent declared DONE without modifying any files")
        else:
            outcome = "hidden_test_failure"
            evidence.append(f"verification exited with failure ({exit_status})")

    # 2. Extract Diagnostic Symptoms (observable empirical flags)
    recs = telemetry_records or []
    max_prompt_eval = max((r.get("prompt_eval_count", 0) for r in recs), default=0)
    output_limit_hit = any(r.get("done_reason") == "length" for r in recs)
    context_full = max_prompt_eval >= num_ctx > 0
    context_near = max_prompt_eval >= (num_ctx * 0.9) > 0
    incomplete_stream = any(r.get("incomplete_stream", False) for r in recs)

    symptoms = {
        "output_limit_reached": output_limit_hit,
        "context_capacity_reached": context_full,
        "context_near_capacity": context_near,
        "invalid_actions_detected": invalid_actions > 0,
        "repeated_command_loop": outcome == "repeated_command",
        "no_files_modified": not changed_files,
        "forbidden_files_modified": bool(forbidden_files),
        "memory_limit_exceeded": outcome == "memory_failure",
        "timeout_exceeded": timed_out,
        "incomplete_stream": incomplete_stream,
        "max_prompt_eval_tokens": max_prompt_eval,
    }

    if output_limit_hit:
        evidence.append("generation truncated due to context/length limit")
    if context_full:
        evidence.append(
            f"prompt tokens ({max_prompt_eval}) reached or exceeded num_ctx ({num_ctx})"
        )
    if incomplete_stream:
        evidence.append("streaming connection closed prematurely before done event")

    # 3. Determine Possible Cause
    # Strictly default to 'unknown' unless conclusive proof establishes the cause.
    # Physical resource exhaustion or repeated invalid tool syntax are conclusive;
    # code failure, test failure, or context pressure do NOT establish cause
    # without controlled ablation.
    cause: str | None = None if outcome == "passed" else "unknown"
    if outcome == "memory_failure":
        cause = "resource_constraint"
        evidence.append("cgroup/memory ceiling breached during execution")
    elif outcome == "invalid_action":
        cause = "tool_format_incompatibility"
        evidence.append("agent failed to adhere to structured interaction format")

    return {
        "outcome": outcome,
        "possible_cause": cause,
        "evidence": evidence,
        "symptoms": symptoms,
    }
