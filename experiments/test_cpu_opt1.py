"""Offline tests for CPU-OPT-1: arm pinning, monitor equivalence (with mutants),
the stratified analysis and the decision rule, and the run's refusal paths."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest import mock

from experiments import cpu_opt1 as opt
from experiments import cpu_profile as cp
from experiments import monitor_equiv as eq

SLEEP = "            time.sleep(self.interval_s)"
WAIT = "            self.stop_event.wait(self.interval_s)"
TELEMETRY = os.path.join(cp.ROOT, "bench", "telemetry.py")


def treatment_source(*, mutate: tuple[str, str] | None = None) -> str:
    with open(TELEMETRY, encoding="utf-8") as f:
        src = f.read()
    assert SLEEP in src
    src = src.replace(SLEEP, WAIT)
    if mutate:
        assert mutate[0] in src
        src = src.replace(*mutate)
    return src


class TestMonitorEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="cpu-opt1-test-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def problems(self, source: str) -> list[str]:
        path = os.path.join(self.tmp, "treatment.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        return eq.equivalence_problems(TELEMETRY, path)

    def test_the_registered_treatment_is_equivalent(self):
        self.assertEqual(self.problems(treatment_source()), [])

    def test_the_control_is_not_its_own_treatment(self):
        with open(TELEMETRY, encoding="utf-8") as f:
            found = self.problems(f.read())
        self.assertTrue(any("kept waiting" in p for p in found), found)

    def test_a_treatment_that_skips_a_query_is_caught(self):
        skip = (
            "metrics = get_gpu_memory()",
            "metrics = get_gpu_memory() if self.sample_count != 1 else {}",
        )
        found = self.problems(treatment_source(mutate=skip))
        self.assertTrue(any("records differ" in p for p in found), found)

    def test_a_treatment_that_stops_sampling_early_is_caught(self):
        early = (
            "self.sample_count += 1",
            "self.sample_count += 1\n"
            "            if self.sample_count >= 2:\n"
            "                return",
        )
        found = self.problems(treatment_source(mutate=early))
        self.assertTrue(any("records differ" in p for p in found), found)

    def test_a_treatment_that_still_sleeps_is_caught(self):
        with open(TELEMETRY, encoding="utf-8") as f:
            found = self.problems(f.read())
        self.assertTrue(found)


class TestArms(unittest.TestCase):
    def setUp(self) -> None:
        self.base = tempfile.mkdtemp(prefix="cpu-opt1-arms-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.control, self.treatment = opt.make_arms(self.base, allow_dirty=True)
        self.addCleanup(opt.drop_arms, self.control, self.treatment)

    def test_the_arms_differ_by_exactly_the_registered_change(self):
        self.assertEqual(opt.arm_problems(self.control, self.treatment), [])

    def test_the_pinned_treatment_hash_matches_the_patch(self):
        got = opt.sha256_file(os.path.join(self.treatment, opt.TELEMETRY))
        self.assertEqual(got, opt.TREATMENT_TELEMETRY)

    def edit(self, root: str, rel: str, old: str, new: str) -> None:
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))

    def test_a_second_change_in_the_treatment_is_refused(self):
        self.edit(
            self.treatment,
            opt.TELEMETRY,
            "interval_s: float = 0.5",
            "interval_s: float = 0.4",
        )
        self.assertTrue(opt.arm_problems(self.control, self.treatment))

    def test_a_change_in_another_file_is_refused(self):
        readme = os.path.join(self.treatment, "README.md")
        with open(readme, "a", encoding="utf-8") as f:
            f.write("x\n")
        found = opt.arm_problems(self.control, self.treatment)
        self.assertTrue(any("more than" in p for p in found), found)

    def test_a_modified_control_is_refused(self):
        self.edit(self.control, opt.TELEMETRY, "0.5", "0.6")
        self.assertTrue(opt.arm_problems(self.control, self.treatment))

    def test_a_different_one_line_change_is_refused(self):
        self.edit(
            self.treatment, opt.TELEMETRY, WAIT.strip(), "self.stop_event.wait(0.1)"
        )
        found = opt.arm_problems(self.control, self.treatment)
        self.assertTrue(any("registered change" in p or "pinned" in p for p in found))

    def test_a_modified_pinned_engine_file_is_refused(self):
        self.edit(
            self.treatment,
            "bench/bench.py",
            "import",
            "import",
        )
        with open(
            os.path.join(self.treatment, "bench/bench.py"), "a", encoding="utf-8"
        ) as f:
            f.write("\n")
        self.assertTrue(opt.arm_problems(self.control, self.treatment))


def row(arm: str, task: str, rep: int, wall: float, **kw: Any) -> dict[str, Any]:
    base = {
        "task": task,
        "arm": arm,
        "rep": rep,
        "wall_s": wall,
        "cpu_s": 0.4,
        "digest": f"digest-{task}",
        "samples": 2,
        "ok": True,
        "problems": [],
        "pressure": 1.0,
    }
    return base | kw


def matrix(gain: float, tasks=("a", "b", "c"), reps: int = 4, **kw: Any):
    rows = []
    for rep in range(reps):
        for i, t in enumerate(tasks):
            noise = 0.01 * ((rep + i) % 3)
            rows.append(row("control", t, rep, 1.5 + noise, **kw))
            rows.append(row("treatment", t, rep, 1.5 + noise - gain, **kw))
    return rows


class TestAnalysis(unittest.TestCase):
    def test_the_bootstrap_resamples_within_tasks(self):
        # 5 pairs of 1.0 in one task and 5 of 0.0 in the other: every stratified
        # resample has exactly five of each, so its median is always 0.5.
        lo, hi = opt.stratified_interval({"x": [1.0] * 5, "y": [0.0] * 5}, n=500)
        self.assertEqual((lo, hi), (0.5, 0.5))

    def test_the_bootstrap_is_deterministic(self):
        data = {"x": [0.4, 0.5, 0.6, 0.55], "y": [0.3, 0.7, 0.5, 0.45]}
        self.assertEqual(
            opt.stratified_interval(data, n=300), opt.stratified_interval(data, n=300)
        )

    def test_a_real_gain_is_adopted(self):
        stats = opt.analyse(matrix(0.45))
        self.assertEqual(opt.decide(stats), "ADOPT AS v0.6 CANDIDATE")

    def test_a_gain_below_the_minimum_is_negative(self):
        stats = opt.analyse(matrix(0.10))
        self.assertTrue(opt.decide(stats).startswith("NEGATIVE"))

    def test_no_gain_is_negative(self):
        self.assertTrue(opt.decide(opt.analyse(matrix(0.0))).startswith("NEGATIVE"))

    def test_a_cpu_regression_blocks_adoption(self):
        rows = matrix(0.45)
        for r in rows:
            if r["arm"] == "treatment":
                r["cpu_s"] = 0.6
        self.assertTrue(opt.decide(opt.analyse(rows)).startswith("NEGATIVE"))

    def test_load_halves_that_disagree_are_inconclusive(self):
        rows = matrix(0.45)
        for r in rows:  # high-pressure pairs show no gain at all
            r["pressure"] = 9.0 if r["rep"] >= 2 else 1.0
            if r["arm"] == "treatment" and r["rep"] >= 2:
                r["wall_s"] += 0.5
        self.assertEqual(opt.decide(opt.analyse(rows)), "INCONCLUSIVE DUE TO LOAD")

    def test_no_load_data_is_inconclusive_not_adopted(self):
        stats = opt.analyse(matrix(0.45, pressure=None))
        self.assertEqual(opt.decide(stats), "INCONCLUSIVE DUE TO LOAD")

    def test_pairs_are_counterbalanced(self):
        firsts = {
            (rep, opt.pair_order(rep, i)[0]) for rep in range(2) for i in range(4)
        }
        self.assertEqual(
            firsts, {(0, "control"), (0, "treatment"), (1, "control"), (1, "treatment")}
        )


class TestValidity(unittest.TestCase):
    tasks = ("a", "b", "c")

    def check(self, rows: list[dict[str, Any]], budget: float = 600.0) -> list[str]:
        return opt.validity_problems(rows, list(self.tasks), 4, budget)

    def test_a_complete_matching_run_is_valid(self):
        self.assertEqual(self.check(matrix(0.4)), [])

    def test_a_failed_attempt_invalidates(self):
        rows = matrix(0.4)
        rows[3] |= {"ok": False, "problems": ["boom"]}
        self.assertTrue(self.check(rows))

    def test_a_digest_that_differs_between_arms_invalidates(self):
        rows = matrix(0.4)
        rows[1]["digest"] = "other"
        self.assertTrue(any("digests" in p for p in self.check(rows)))

    def test_a_missing_digest_invalidates(self):
        rows = matrix(0.4)
        rows[0]["digest"] = None
        self.assertTrue(any("digests" in p for p in self.check(rows)))

    def test_a_short_matrix_invalidates(self):
        self.assertTrue(any("incomplete" in p for p in self.check(matrix(0.4)[:-2])))

    def test_an_overshot_budget_invalidates(self):
        self.assertTrue(any("budget" in p for p in self.check(matrix(0.4), budget=1.0)))

    def test_sample_counts_that_differ_between_arms_invalidate(self):
        rows = matrix(0.4)
        for r in rows:
            if r["arm"] == "treatment":
                r["samples"] = 1  # a monitor that quietly takes fewer samples
        self.assertTrue(any("sample counts" in p for p in self.check(rows)))

    def test_one_sample_of_jitter_is_tolerated_but_not_in_the_median(self):
        rows = matrix(0.4)
        rows[1]["samples"] = 1  # a single pair off by one
        self.assertEqual(self.check(rows), [])

    def test_a_monitor_that_never_ran_is_a_failed_attempt(self):
        done = ({"wall_s": 1.0, "user_s": 0.1, "sys_s": 0.1, "maxrss_kib": 1}, 0, False)
        with (
            mock.patch.object(
                opt.cp, "read_row", lambda d: {"monitor_sample_count": 0}
            ),
            mock.patch.object(opt.cp, "measure_command", lambda *a, **k: done),
        ):
            got = opt.attempt(
                "control", "/x", "a", opt.Context({}, tempfile.gettempdir(), True)
            )
        self.assertTrue(any("did not sample" in p for p in got["problems"]))
        self.assertFalse(got["ok"])


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.mkdtemp(prefix="cpu-opt1-cli-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = os.path.join(self.tmp, "o")

    def run_with(self, rows: list[dict[str, Any]], *argv: str, problems=()) -> int:
        patches = [
            mock.patch.object(opt, "make_arms", lambda base: ("c", "t")),
            mock.patch.object(opt, "arm_problems", lambda c, t: list(problems)),
            mock.patch.object(opt, "drop_arms", lambda *a: None),
            mock.patch.object(
                opt.monitor_equiv, "equivalence_problems", lambda c, t: []
            ),
            mock.patch.object(opt.cp, "build_references", lambda tasks: ({}, [])),
            mock.patch.object(opt, "measure", lambda *a, **k: (rows, None)),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return opt.main(["run", "--out", self.out, *argv])

    def summary(self) -> dict[str, Any]:
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            return json.load(f)

    def full(self, tasks: list[str], reps: int, gain: float):
        return matrix(gain, tasks=tuple(tasks), reps=reps)

    def test_the_final_set_needs_the_explicit_flag(self):
        self.assertEqual(opt.main(["run", "--out", self.out, "--set", "final"]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out, "summary.json")))

    def test_the_final_run_uses_exactly_the_registered_repetitions(self):
        argv = ["--set", "final", "--registered-final", "--reps", "3"]
        self.assertEqual(opt.main(["run", "--out", self.out, *argv]), 2)

    def test_an_existing_result_is_never_overwritten(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump({"keep": True}, f)
        self.assertEqual(opt.main(["run", "--out", self.out]), 2)
        self.assertEqual(self.summary(), {"keep": True})

    def test_an_invalid_run_has_no_effect_statistics_or_decision(self):
        tasks = cp.workload()["calibration"]
        rows = self.full(tasks, 2, 0.4)
        rows[0] |= {"ok": False, "problems": ["boom"]}
        code = self.run_with(rows, "--reps", "2")
        doc = self.summary()
        self.assertEqual(code, 1)
        self.assertFalse(doc["valid"])
        self.assertNotIn("stats", doc)
        self.assertNotIn("decision", doc)

    def test_an_arm_problem_stops_before_measuring(self):
        measured = mock.Mock(return_value=([], None))
        with mock.patch.object(opt, "measure", measured):
            code = self.run_with([], "--reps", "1", problems=["arms differ"])
        self.assertEqual(code, 1)
        self.assertIn("arms differ", self.summary()["problems"])

    def test_a_calibration_dry_run_makes_no_decision(self):
        tasks = cp.workload()["calibration"]
        code = self.run_with(self.full(tasks, 2, 0.45), "--reps", "2")
        doc = self.summary()
        self.assertEqual(code, 0)
        self.assertTrue(doc["dry_run"])
        self.assertTrue(doc["decision"].startswith("DRY RUN"))

    def test_a_valid_final_run_gets_the_registered_decision(self):
        tasks = cp.workload()["final"]
        code = self.run_with(
            self.full(tasks, opt.REPS, 0.45), "--set", "final", "--registered-final"
        )
        doc = self.summary()
        self.assertEqual(code, 0)
        self.assertFalse(doc["dry_run"])
        self.assertEqual(doc["decision"], "ADOPT AS v0.6 CANDIDATE")


class TestAdmissionControl(unittest.TestCase):
    def test_no_pair_starts_once_the_budget_would_be_exceeded(self):
        calls = []

        def fake(arm: str, root: str, task: str, *a: Any) -> dict[str, Any]:
            calls.append((arm, task))
            return row(arm, task, 0, 1.0, cpu_s=10.0)

        args = mock.Mock(reps=3, cpu_budget=45.0, no_strict=False)
        tmp = tempfile.mkdtemp(prefix="cpu-opt1-adm-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with (
            mock.patch.object(opt, "attempt", fake),
            mock.patch.object(opt, "pressure", lambda: 1.0),
        ):
            rows, stopped = opt.measure(("c", "t"), ["a", "b", "c"], {}, tmp, args)
        self.assertIn("stopped admitting", stopped)
        self.assertEqual(len(rows) % 2, 0)  # a started pair always completes
        self.assertLess(len(rows), 18)


if __name__ == "__main__":
    unittest.main()
