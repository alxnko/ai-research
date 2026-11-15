"""Offline tests of the CPU measurement tooling (no model, no GPU compute)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from typing import Any, ClassVar
from unittest import mock

from bench import bench as engine
from experiments import cpu_profile as cp

TASK = "debugging-off-by-one"


class Quiet(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.mkdtemp(
            prefix="cpu-profile-test-",
            dir="/var/tmp" if os.path.isdir("/var/tmp") else None,
        )
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class TestWorkload(unittest.TestCase):
    def test_the_split_is_disjoint_and_covers_the_suite_exactly(self):
        doc, suite = cp.workload(), engine.load_suite("v0.5")
        self.assertEqual(cp.workload_problems(doc, suite), [])

    def test_problems_are_reported(self):
        suite = {"tasks": {"a": "x", "b": "y", "c": "z"}}
        bad = {"calibration": ["a", "b"], "final": ["b", "c"]}
        self.assertTrue(any("overlap" in p for p in cp.workload_problems(bad, suite)))
        missing = {"calibration": ["a"], "final": ["b"]}
        self.assertTrue(cp.workload_problems(missing, suite))
        dup = {"calibration": ["a", "a"], "final": ["b", "c"]}
        self.assertTrue(any("duplicate" in p for p in cp.workload_problems(dup, suite)))

    def test_every_category_appears_in_both_sets(self):
        def cats(names):
            return {n.split("-")[0].replace("debugging", "debug") for n in names}

        doc = cp.workload()
        self.assertEqual(cats(doc["calibration"]), cats(doc["final"]))


class TestStatistics(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        values = [float(i) for i in range(1, 101)]
        self.assertEqual(cp.nearest_rank(values, 0.95), 95.0)
        self.assertEqual(cp.nearest_rank(values, 0.5), 50.0)
        self.assertEqual(cp.nearest_rank([7.0], 0.95), 7.0)
        self.assertEqual(cp.nearest_rank([3.0, 1.0, 2.0], 1.0), 3.0)

    def test_summary(self):
        s = cp.summarize([1.0, 2.0, 3.0, 10.0])
        self.assertEqual((s["n"], s["median"], s["min"], s["max"]), (4, 2.5, 1.0, 10.0))


class TestCanonicalDigest(unittest.TestCase):
    ROW: ClassVar[dict[str, Any]] = {
        "task": "t",
        "task_digest": "d",
        "passed": True,
        "run_successful": True,
        "outcome": "passed",
        "verification_passed": True,
        "changed_files": ["a.py"],
        "commands_executed": 1,
        "model_turns": 2,
        "tool_calls": 2,
        "invalid_actions": 0,
        "tokens": 140,
    }

    def test_timing_never_changes_it(self):
        slower = {**self.ROW, "total_wall_s": 99.0, "monitor_overhead_s": 5.0}
        self.assertEqual(cp.canonical_digest(slower), cp.canonical_digest(self.ROW))

    def test_every_deterministic_output_changes_it(self):
        base = cp.canonical_digest(self.ROW)
        changes = {
            "passed": False,
            "outcome": "failed",
            "changed_files": [],
            "commands_executed": 0,
            "tokens": 1,
            "verification_passed": False,
            "task_digest": "other",
        }
        for key, value in changes.items():
            self.assertNotEqual(
                cp.canonical_digest({**self.ROW, key: value}), base, key
            )


class TestIsOk(unittest.TestCase):
    def test_only_a_solved_attempt_counts(self):
        good = {"passed": True, "run_successful": True}
        self.assertTrue(cp.is_ok(good))
        for bad in (
            None,
            {},
            {"passed": False, "run_successful": True},
            {"passed": True, "run_successful": False},
            {"passed": True},
            {"passed": "yes", "run_successful": True},
        ):
            self.assertFalse(cp.is_ok(bad), bad)


class TestScriptedActor(Quiet):
    def test_the_command_reproduces_the_reference_solution(self):
        tdir = os.path.join(engine.TASKS, TASK)
        work = os.path.join(self.tmp, "work")
        os.makedirs(work)
        done = subprocess.run(
            ["sh", "-c", cp.solution_command(tdir)], cwd=work, capture_output=True
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        sol = os.path.join(tdir, "solution")
        for name in os.listdir(sol):
            want = pathlib.Path(sol, name).read_bytes()
            self.assertEqual(pathlib.Path(work, name).read_bytes(), want, name)

    def test_nested_solution_files_are_created_with_their_directories(self):
        tdir = os.path.join(self.tmp, "task")
        os.makedirs(os.path.join(tdir, "solution", "pkg", "sub"))
        with open(os.path.join(tdir, "solution", "pkg", "sub", "m.py"), "w") as f:
            f.write("x = 'q'\n")
        work = os.path.join(self.tmp, "w")
        os.makedirs(work)
        subprocess.run(["sh", "-c", cp.solution_command(tdir)], cwd=work, check=True)
        with open(os.path.join(work, "pkg", "sub", "m.py")) as f:
            self.assertEqual(f.read(), "x = 'q'\n")


class TestRealRun(Quiet):
    """The unchanged engine, end to end (non-strict: CI may lack cgroups)."""

    def test_a_scripted_attempt_passes_and_is_reproducible(self):
        first = cp.record(self.plan(), TASK, 0, self.tmp)
        second = cp.record(self.plan(), TASK, 1, self.tmp)
        for row in (first, second):
            self.assertTrue(row["ok"], row)
            self.assertEqual(row["exit"], 0)
            self.assertGreater(row["wall_s"], 0)
            self.assertGreater(row["maxrss_kib"], 0)
            self.assertAlmostEqual(row["cpu_s"], row["user_s"] + row["sys_s"])
            # one scripted command, then finish: the shape of the workflow
            self.assertEqual((row["commands_executed"], row["model_turns"]), (1, 2))
        self.assertEqual(first["digest"], second["digest"])
        self.assertTrue(first["digest"])

    def plan(self) -> cp.Plan:
        return cp.Plan("calibration", [TASK], 1, 1, False, 100.0)

    def test_an_actor_that_does_not_solve_the_task_is_a_failure(self):
        # the correctness gate is not vacuous: skipping the work is detected
        run_dir = os.path.join(self.tmp, "idle")
        with (
            mock.patch.object(cp, "solution_command", lambda d: "true"),
            mock.patch.dict(os.environ, {"BENCH_MOCK_MODEL": "1"}),
        ):
            cp.one_main(["--task", TASK, "--run-dir", run_dir, "--no-strict-isolation"])
        row = cp.read_row(run_dir)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertFalse(row["passed"])
        good = cp.canonical_digest({**row, "passed": True})
        self.assertNotEqual(cp.canonical_digest(row), good)

    def test_a_child_that_crashes_is_recorded_as_a_failure(self):
        with mock.patch.object(
            cp,
            "measure_child",
            lambda *a, **k: (
                {"wall_s": 1.0, "user_s": 0.1, "sys_s": 0.1, "maxrss_kib": 1},
                1,
            ),
        ):
            row = cp.record(self.plan(), TASK, 0, self.tmp)
        self.assertFalse(row["ok"])
        self.assertIsNone(row["digest"])


class TestBatches(Quiet):
    @staticmethod
    def fake(cpu: float = 1.0, ok: bool = True):
        def record(plan, task, rep, scratch):
            return {
                "set": plan.label, "task": task, "rep": rep, "workers": plan.workers,
                "wall_s": cpu, "user_s": cpu, "sys_s": 0.0, "cpu_s": cpu,
                "maxrss_kib": 1, "exit": 0, "ok": ok, "digest": "d",
                "engine_total_wall_s": None, "agent_wall_s": None,
                "verification_wall_s": None, "monitor_overhead_s": None,
            }  # fmt: skip

        return record

    def test_workers_are_capped(self):
        with mock.patch.object(cp, "record", self.fake()):
            rows, batch = cp.run_batches(
                cp.Plan("calibration", ["a", "b"], 1, 99, False, 100.0)
            )
        self.assertEqual(batch["workers"], cp.MAX_WORKERS)
        self.assertEqual(len(rows), 2)

    def test_the_cpu_budget_stops_the_run_and_says_so(self):
        with mock.patch.object(cp, "record", self.fake(cpu=10.0)):
            rows, batch = cp.run_batches(
                cp.Plan("calibration", list("abcdef"), 1, 1, False, 25.0)
            )
        self.assertEqual(len(rows), 3)  # 10 + 10 + 10 >= 25 stops before a fourth
        self.assertEqual(batch["attempted"], 3)
        self.assertEqual(batch["planned"], 6)
        self.assertIn("budget", batch["stopped"])

    def test_cpu_already_spent_counts_against_the_budget(self):
        with mock.patch.object(cp, "record", self.fake(cpu=1.0)):
            rows, batch = cp.run_batches(
                cp.Plan("calibration", ["a"], 1, 1, False, 20.0, spent_s=20.0)
            )
        self.assertEqual(rows, [])
        self.assertIsNotNone(batch["stopped"])

    def test_an_exhausted_budget_is_never_reported_as_complete(self):
        with mock.patch.object(cp, "record", self.fake(cpu=30.0)):
            _rows, batch = cp.run_batches(
                cp.Plan("calibration", ["a", "b"], 1, 1, False, 20.0)
            )
        self.assertLess(batch["attempted"], batch["planned"])

    def test_summary_reports_failures_and_nondeterminism(self):
        a = self.fake()(cp.Plan("calibration", [], 1, 1, False, 1.0), "t", 0, "")
        b = {**a, "rep": 1, "digest": "other"}
        c = {**a, "task": "u", "ok": False, "digest": None}
        out = cp.summary([a, b, c], [])
        group = out["groups"]["calibration/workers=1"]
        self.assertEqual(group["execution_failures"], 1)
        self.assertEqual(group["nondeterministic_tasks"], ["t", "u"])
        self.assertEqual(cp.digests([a, b, c]), {})  # no single reference digest


class TestCli(Quiet):
    def run_cli(self, *argv: str) -> int:
        return cp.main(["run", *argv])

    def test_the_final_set_needs_an_explicit_flag(self):
        out = os.path.join(self.tmp, "o")
        self.assertEqual(self.run_cli("--set", "final", "--out", out), 2)
        self.assertFalse(os.path.exists(os.path.join(out, "measurements.jsonl")))

    def test_an_existing_result_is_never_overwritten(self):
        out = os.path.join(self.tmp, "o")
        os.makedirs(out)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"keep": True}, f)
        self.assertEqual(self.run_cli("--set", "calibration", "--out", out), 2)
        with open(os.path.join(out, "summary.json")) as f:
            self.assertEqual(json.load(f), {"keep": True})

    def test_a_run_writes_raw_measurements_and_a_summary(self):
        out = os.path.join(self.tmp, "o")
        fake = TestBatches.fake()
        with (
            mock.patch.object(cp, "record", fake),
            mock.patch.object(
                cp, "workload", lambda: {"calibration": ["a", "b"], "final": []}
            ),
        ):
            code = self.run_cli(
                "--set", "calibration", "--reps", "2", "--workers", "1", "2",
                "--out", out, "--no-strict-isolation",
            )  # fmt: skip
        self.assertEqual(code, 0)
        with open(os.path.join(out, "measurements.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 8)  # 2 tasks x 2 reps x 2 settings
        with open(os.path.join(out, "summary.json")) as f:
            doc = json.load(f)
        self.assertEqual(
            sorted(doc["groups"]), ["calibration/workers=1", "calibration/workers=2"]
        )
        self.assertIn("workload_sha256", doc)
        self.assertEqual(cp.main(["report", out]), 0)

    def test_a_failed_attempt_makes_the_run_exit_non_zero(self):
        out = os.path.join(self.tmp, "o")
        with (
            mock.patch.object(cp, "record", TestBatches.fake(ok=False)),
            mock.patch.object(
                cp, "workload", lambda: {"calibration": ["a"], "final": []}
            ),
        ):
            code = self.run_cli("--set", "calibration", "--reps", "1", "--out", out)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
