"""Offline tests for CPU-OPT-1: arm pinning, monitor equivalence (with mutants),
the stratified analysis and the decision rule, and the run's refusal paths."""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from experiments import cpu_opt1 as opt
from experiments import cpu_profile as cp
from experiments import monitor_equiv as eq

SLEEP = "            time.sleep(self.interval_s)"
WAIT = "            self.stop_event.wait(self.interval_s)"
TELEMETRY = os.path.join(cp.ROOT, "bench", "telemetry.py")  # the checkout (v0.6)


def v05_source() -> bytes:
    """The control: `bench/telemetry.py` as released in the immutable v0.5 tag."""
    return subprocess.run(
        ["git", "-C", cp.ROOT, "show", "v0.5:bench/telemetry.py"],
        capture_output=True,
        check=True,
    ).stdout


def _control_file() -> str:
    folder = tempfile.mkdtemp(prefix="cpu-opt1-v05-", dir="/var/tmp")
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    path = os.path.join(folder, "telemetry.py")
    with open(path, "wb") as f:
        f.write(v05_source())
    return path


CONTROL = _control_file()


def treatment_source(*, mutate: tuple[str, str] | None = None) -> str:
    with open(CONTROL, encoding="utf-8") as f:
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
        return eq.equivalence_problems(CONTROL, path)

    def test_the_registered_treatment_is_equivalent(self):
        self.assertEqual(self.problems(treatment_source()), [])

    def test_the_control_is_not_its_own_treatment(self):
        with open(CONTROL, encoding="utf-8") as f:
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
        with open(CONTROL, encoding="utf-8") as f:
            found = self.problems(f.read())
        self.assertTrue(found)


class TestArms(unittest.TestCase):
    def setUp(self) -> None:
        self.base = tempfile.mkdtemp(prefix="cpu-opt1-arms-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        # The runner builds arms from a commit whose telemetry is the v0.5 control.
        # The checkout is v0.6, so reproduce that state in a throwaway clone.
        clone = os.path.join(self.base, "clone")
        run = ["git", "-C", clone]
        subprocess.run(
            ["git", "clone", "-q", "--no-hardlinks", cp.ROOT, clone], check=True
        )
        with open(os.path.join(clone, opt.TELEMETRY), "wb") as f:
            f.write(v05_source())
        who = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run([*run, *who, "commit", "-qam", "v0.5 telemetry"], check=True)
        for target, value in (
            (cp, ("ROOT", clone)),
            (
                opt,
                (
                    "PATCH",
                    os.path.join(clone, "experiments", "cpu_opt1_treatment.patch"),
                ),
            ),
        ):
            self.enterContext(mock.patch.object(target, *value))
        self.control, self.treatment = opt.make_arms(self.base, allow_dirty=True)
        self.addCleanup(opt.drop_arms, self.control, self.treatment)

    def test_the_arms_differ_by_exactly_the_registered_change(self):
        self.assertEqual(opt.arm_problems(self.control, self.treatment), [])

    def test_the_release_checkout_is_the_validated_treatment(self):
        # v0.6 is exactly the control plus the registered patch
        self.assertEqual(opt.sha256_file(TELEMETRY), opt.TREATMENT_TELEMETRY)
        self.assertEqual(opt.sha256_file(CONTROL), opt.CONTROL_TELEMETRY)

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
            kws = {"pressure": 1.0 + 4.0 * (rep % 2)} | kw
            rows.append(row("control", t, rep, 1.5 + noise, **kws))
            rows.append(row("treatment", t, rep, 1.5 + noise - gain, **kws))
    return rows


class TestRegisteredSettings(unittest.TestCase):
    def test_the_fixed_parameters_are_the_registered_ones(self):
        self.assertEqual(
            (opt.MIN_HALF, opt.CPU_BUDGET_S, opt.REPS, opt.SEED, opt.RESAMPLES),
            (20, 600.0, 10, 20261118, 10_000),
        )
        self.assertEqual(
            (opt.MIN_IMPROVEMENT_S, opt.MIN_LOWER_BOUND_S, opt.MAX_CPU_REGRESSION_S),
            (0.20, 0.10, 0.02),
        )


class TestAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(mock.patch.object(opt, "MIN_HALF", 2))

    def test_an_idle_host_makes_the_load_groups_not_evaluable(self):
        stats = opt.analyse(matrix(0.45, pressure=0.0))  # avg10 is 0.00 throughout
        self.assertIsNone(stats["load_halves"])
        self.assertEqual(opt.decide(stats), "INCONCLUSIVE DUE TO LOAD")

    def test_pairs_at_the_cut_belong_to_neither_group(self):
        rows = matrix(0.45)  # 12 pairs, three per repetition
        for r in rows:
            r["pressure"] = {0: 1.0, 1: 3.0, 2: 3.0, 3: 5.0}[r["rep"]]  # median 3.0
        halves = opt.analyse(rows)["load_halves"]
        self.assertEqual((halves["low_pairs"], halves["high_pairs"]), (3, 3))

    def test_a_group_below_the_minimum_size_is_not_evaluable(self):
        with mock.patch.object(opt, "MIN_HALF", 7):
            self.assertIsNone(opt.analyse(matrix(0.45))["load_halves"])

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
        got = self.check(matrix(0.4)[:-2])
        self.assertTrue(any("missing cells" in p for p in got), got)

    def test_a_lost_pair_hidden_by_a_duplicate_is_caught(self):
        rows = matrix(0.4)
        del rows[0:2]  # the whole (a, rep 0) pair
        rows += [dict(rows[0]), dict(rows[1])]  # same row count, wrong matrix
        self.assertEqual(len(rows), len(matrix(0.4)))
        got = self.check(rows)
        self.assertTrue(any("missing cells" in p for p in got), got)
        self.assertTrue(any("duplicate cells" in p for p in got), got)

    def test_an_unexpected_cell_is_caught(self):
        for change in ({"rep": 99}, {"task": "zzz"}, {"arm": "third"}):
            rows = matrix(0.4)
            rows[0] |= change
            got = self.check(rows)
            with self.subTest(change=change):
                self.assertTrue(any("unexpected cells" in p for p in got), got)

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
        # never touch the real reservation under runs/
        reservation = os.path.join(self.tmp, "reservation")
        self.enterContext(mock.patch.object(opt, "RESERVATION_DIR", reservation))
        self.final = cp.workload()["final"]

    def run_with(
        self,
        rows: list[dict[str, Any]],
        *argv: str,
        out: str | None = None,
        problems=(),
        measure=None,
    ) -> int:
        def fake_measure(arms, tasks, ctx, args, record):
            for r in rows:
                record(r)

        patches = [
            mock.patch.object(opt, "make_arms", lambda base: ("c", "t")),
            mock.patch.object(opt, "arm_problems", lambda c, t: list(problems)),
            mock.patch.object(opt, "drop_arms", lambda *a: None),
            mock.patch.object(
                opt.monitor_equiv, "equivalence_problems", lambda c, t: []
            ),
            mock.patch.object(opt.cp, "build_references", lambda tasks: ({}, [])),
            mock.patch.object(opt, "measure", measure or fake_measure),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return opt.main(["run", "--out", out or self.out, *argv])

    def summary(self, out: str | None = None) -> dict[str, Any]:
        with open(os.path.join(out or self.out, "summary.json"), encoding="utf-8") as f:
            return json.load(f)

    def good_final(self, gain: float = 0.45) -> list[dict[str, Any]]:
        return matrix(gain, tasks=tuple(self.final), reps=opt.REPS)

    FINAL = ("--set", "final", "--registered-final")

    # -- pinned final-run settings
    def test_the_final_set_needs_the_explicit_flag(self):
        self.assertEqual(opt.main(["run", "--out", self.out, "--set", "final"]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out, "summary.json")))

    def test_the_final_run_uses_exactly_the_registered_repetitions(self):
        argv = [*self.FINAL, "--reps", "3"]
        self.assertEqual(opt.main(["run", "--out", self.out, *argv]), 2)

    def test_the_final_run_refuses_relaxed_isolation(self):
        argv = [*self.FINAL, "--no-strict"]
        self.assertEqual(opt.main(["run", "--out", self.out, *argv]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out, "attempts.jsonl")))

    def test_the_final_run_refuses_another_cpu_budget(self):
        for budget in ("300", "1200", "599.9"):
            with self.subTest(budget=budget):
                argv = [*self.FINAL, "--cpu-budget", budget]
                self.assertEqual(opt.main(["run", "--out", self.out, *argv]), 2)

    def test_the_dry_run_still_allows_its_own_settings(self):
        rows = matrix(0.4, tasks=tuple(cp.workload()["calibration"]), reps=2)
        code = self.run_with(rows, "--reps", "2", "--no-strict", "--cpu-budget", "50")
        self.assertEqual(code, 0)

    def test_an_existing_result_is_never_overwritten(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump({"keep": True}, f)
        self.assertEqual(opt.main(["run", "--out", self.out]), 2)
        self.assertEqual(self.summary(), {"keep": True})

    def test_partial_records_are_never_written_over(self):
        os.makedirs(self.out)
        path = os.path.join(self.out, "attempts.jsonl")
        with open(path, "w") as f:
            f.write("{}\n")
        self.assertEqual(opt.main(["run", "--out", self.out]), 2)
        with open(path) as f:
            self.assertEqual(f.read(), "{}\n")

    # -- validity and decision
    def test_an_invalid_run_has_no_effect_statistics_or_decision(self):
        tasks = cp.workload()["calibration"]
        rows = matrix(0.4, tasks=tuple(tasks), reps=2)
        rows[0] |= {"ok": False, "problems": ["boom"]}
        code = self.run_with(rows, "--reps", "2")
        doc = self.summary()
        self.assertEqual(code, 1)
        self.assertFalse(doc["valid"])
        self.assertNotIn("stats", doc)
        self.assertNotIn("decision", doc)

    def test_a_wrong_matrix_with_the_right_row_count_is_invalid(self):
        tasks = tuple(cp.workload()["calibration"])
        rows = matrix(0.4, tasks=tasks, reps=2)
        del rows[0:2]
        rows += [dict(rows[0]), dict(rows[1])]
        code = self.run_with(rows, "--reps", "2")
        self.assertEqual(code, 1)
        self.assertNotIn("stats", self.summary())

    def test_an_arm_problem_stops_before_measuring(self):
        called = mock.Mock()
        code = self.run_with(
            [], "--reps", "1", problems=["arms differ"], measure=called
        )
        self.assertEqual(code, 1)
        called.assert_not_called()
        self.assertIn("arms differ", self.summary()["problems"])

    def test_a_calibration_dry_run_makes_no_decision(self):
        tasks = cp.workload()["calibration"]
        rows = matrix(0.45, tasks=tuple(tasks), reps=2)
        code = self.run_with(rows, "--reps", "2")
        doc = self.summary()
        self.assertEqual(code, 0)
        self.assertTrue(doc["dry_run"])
        self.assertTrue(doc["decision"].startswith("DRY RUN"))

    def test_a_valid_final_run_gets_the_registered_decision(self):
        code = self.run_with(self.good_final(), *self.FINAL)
        doc = self.summary()
        self.assertEqual(code, 0)
        self.assertFalse(doc["dry_run"])
        self.assertEqual(doc["decision"], "ADOPT AS v0.6 CANDIDATE")

    # -- crash safety
    def test_completed_attempts_survive_a_crash_mid_run(self):
        calls = []

        def attempt(arm: str, root: str, task: str, ctx: Any) -> dict[str, Any]:
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("host fault")
            return row(arm, task, 0, 1.0)

        patches = [
            mock.patch.object(opt, "attempt", attempt),
            mock.patch.object(opt, "pressure", lambda: 1.0),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            code = self.run_with([], "--reps", "1", measure=opt.measure)
        with open(os.path.join(self.out, "attempts.jsonl")) as f:
            self.assertEqual(len(f.read().splitlines()), 2)
        doc = self.summary()
        self.assertEqual(code, 1)
        self.assertIn("host fault", doc["aborted"])
        self.assertNotIn("stats", doc)

    def test_a_failed_attempt_stops_the_run_and_keeps_its_log(self):
        log = os.path.join(self.tmp, "child.log")
        with open(log, "w") as f:
            f.write("child output")
        calls = []

        def attempt(arm: str, root: str, task: str, ctx: Any) -> dict[str, Any]:
            calls.append(arm)
            bad = len(calls) == 2
            return row(
                arm,
                task,
                0,
                1.0,
                ok=not bad,
                problems=["boom"] if bad else [],
                failed_log=log if bad else None,
            )

        with (
            mock.patch.object(opt, "attempt", attempt),
            mock.patch.object(opt, "pressure", lambda: 1.0),
        ):
            code = self.run_with([], "--reps", "2", measure=opt.measure)
        self.assertEqual(len(calls), 2)  # nothing after the failed attempt
        self.assertEqual(code, 1)
        doc = self.summary()
        self.assertTrue(any("failed attempt" in p for p in doc["problems"]))
        self.assertNotIn("stats", doc)
        with open(os.path.join(self.out, "failed", "child.log")) as f:
            self.assertEqual(f.read(), "child output")

    # -- one-shot final run and recovery
    def second(self) -> str:
        return os.path.join(self.tmp, "second")

    def test_a_second_final_run_is_refused_even_with_another_out_dir(self):
        self.assertEqual(self.run_with(self.good_final(), *self.FINAL), 0)
        measured = mock.Mock()
        code = self.run_with(
            self.good_final(), *self.FINAL, out=self.second(), measure=measured
        )
        self.assertEqual(code, 1)
        measured.assert_not_called()
        got = self.summary(self.second())["problems"]
        self.assertTrue(any("already started" in p for p in got), got)

    def test_a_failed_preflight_does_not_consume_the_final_run(self):
        code = self.run_with([], *self.FINAL, problems=["arms differ"])
        self.assertEqual(code, 1)
        self.assertFalse(
            os.path.exists(os.path.join(opt.RESERVATION_DIR, opt.RESERVED))
        )
        self.assertEqual(
            self.run_with(self.good_final(), *self.FINAL, out=self.second()), 0
        )

    def crashed_final(self) -> None:
        def boom(arms, tasks, ctx, args, record):
            raise RuntimeError("host fault")

        self.assertEqual(self.run_with([], *self.FINAL, measure=boom), 1)

    def test_an_aborted_final_run_may_be_recovered_exactly_once(self):
        self.crashed_final()
        recover = [*self.FINAL, "--recover-of", self.out]
        code = self.run_with(self.good_final(), *recover, out=self.second())
        doc = self.summary(self.second())
        self.assertEqual(code, 0)
        self.assertEqual(doc["recovery_of"], self.out)
        self.assertEqual(doc["decision"], "ADOPT AS v0.6 CANDIDATE")
        self.assertTrue(os.path.exists(os.path.join(self.out, "summary.json")))
        third = os.path.join(self.tmp, "third")
        code = self.run_with(self.good_final(), *recover, out=third)
        self.assertEqual(code, 1)
        got = self.summary(third)["problems"]
        self.assertTrue(any("already used" in p for p in got), got)

    def test_a_killed_final_run_may_be_recovered(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "started.json"), "w") as f:
            f.write("{}")
        os.makedirs(opt.RESERVATION_DIR)
        with open(os.path.join(opt.RESERVATION_DIR, opt.RESERVED), "w") as f:
            json.dump({"out": self.out}, f)
        self.assertIsNone(opt.recovery_problem(self.out))

    def test_a_gate_failure_is_a_result_not_an_infrastructure_failure(self):
        rows = self.good_final()
        rows[0] |= {"ok": False, "problems": ["digest mismatch"]}
        self.assertEqual(self.run_with(rows, *self.FINAL), 1)
        got = opt.recovery_problem(self.out)
        self.assertIn("failed a gate", got or "")

    def test_a_completed_final_run_cannot_be_rerun(self):
        self.assertEqual(self.run_with(self.good_final(), *self.FINAL), 0)
        self.assertIn("produced a result", opt.recovery_problem(self.out) or "")

    def test_recovery_must_name_the_run_that_holds_the_reservation(self):
        self.crashed_final()
        self.assertIn("reservation", opt.recovery_problem(self.second()) or "")

    def test_recovery_is_only_for_the_final_run(self):
        self.assertEqual(
            opt.main(["run", "--out", self.out, "--recover-of", self.tmp]), 2
        )


class TestAdmissionControl(unittest.TestCase):
    def test_no_pair_starts_once_the_budget_would_be_exceeded(self):
        def fake(arm: str, root: str, task: str, ctx: Any) -> dict[str, Any]:
            return row(arm, task, 0, 1.0, cpu_s=10.0)

        args = mock.Mock(reps=3, cpu_budget=45.0, no_strict=False)
        tmp = tempfile.mkdtemp(prefix="cpu-opt1-adm-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rec = opt.Recorder(tmp)
        with (
            mock.patch.object(opt, "attempt", fake),
            mock.patch.object(opt, "pressure", lambda: 1.0),
        ):
            stopped = opt.measure(("c", "t"), ["a", "b", "c"], None, args, rec)
        self.assertIn("stopped admitting", stopped or "")
        self.assertEqual(len(rec.rows) % 2, 0)  # a started pair always completes
        self.assertLess(len(rec.rows), 18)


class TestEquivalenceBarrier(unittest.TestCase):
    """The real-thread part synchronizes at the wait boundary, both ways."""

    def test_in_flight_and_ordinary_shutdown_are_both_exercised(self):
        control = eq.load(CONTROL, "opt1_test_control")
        self.addCleanup(sys.modules.pop, "opt1_test_control", None)
        waited, record = eq.real_stop(control, in_flight=True)
        self.assertGreaterEqual(waited, 0.75 * eq.IN_FLIGHT_S)
        self.assertEqual(record["monitor_sample_count"], 1)
        ordinary, _ = eq.real_stop(control, in_flight=False)
        self.assertGreater(ordinary, 5 * eq.STOP_BOUND_S)


if __name__ == "__main__":
    unittest.main()
