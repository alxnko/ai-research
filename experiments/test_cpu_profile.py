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
from collections.abc import Callable
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


def good_row() -> dict[str, Any]:
    return {
        "task": "t",
        "task_digest": "d",
        "passed": True,
        "run_successful": True,
        "outcome": "passed",
        "verification_passed": True,
        "changed_files": ["a.py"],
        "added_files": [],
        "removed_files": [],
        "forbidden_files": [],
        "commands_executed": 1,
        "model_turns": 2,
        "tool_calls": 2,
        "invalid_actions": 0,
        "tokens": 140,
        "work_sha256": "w",
        "hidden_sha256": "h",
        "verification_evidence_sha256": "e",
    }


def fake_record(
    cpu: float = 1.0,
    ok: bool = True,
    digest: Callable[[Any, str], str | None] = lambda plan, task: "d",
):
    def record(plan, task, rep, scratch):
        return {
            "set": plan.label, "task": task, "rep": rep, "workers": plan.workers,
            "wall_s": cpu, "user_s": cpu, "sys_s": 0.0, "cpu_s": cpu,
            "maxrss_kib": 1, "exit": 0, "timed_out": False, "ok": ok,
            "problems": [] if ok else ["boom"], "digest": digest(plan, task),
            "engine_total_wall_s": None, "agent_wall_s": None,
            "verification_wall_s": None, "monitor_overhead_s": None,
        }  # fmt: skip

    return record


def plan(**kw: Any) -> cp.Plan:
    base: dict[str, Any] = {
        "label": "calibration",
        "tasks": ["a", "b"],
        "reps": 1,
        "workers": 1,
        "strict": False,
        "budget_s": 100.0,
    }
    return cp.Plan(**{**base, **kw})


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
    def test_timing_never_changes_it(self):
        slower = {**good_row(), "total_wall_s": 99.0, "monitor_overhead_s": 5.0}
        self.assertEqual(cp.canonical_digest(slower), cp.canonical_digest(good_row()))

    def test_every_deterministic_field_changes_it(self):
        base = cp.canonical_digest(good_row())
        self.assertIsNotNone(base)
        for key in cp.DETERMINISTIC:
            changed = {**good_row(), key: "changed"}
            self.assertNotEqual(cp.canonical_digest(changed), base, key)

    def test_a_missing_or_null_field_gives_no_digest_instead_of_hashing_none(self):
        for key in cp.DETERMINISTIC:
            absent = {k: v for k, v in good_row().items() if k != key}
            self.assertIsNone(cp.canonical_digest(absent), key)
            self.assertEqual(cp.missing_fields(absent), [key])
            self.assertIsNone(cp.canonical_digest({**good_row(), key: None}), key)

    def test_the_disk_measured_fields_are_part_of_the_digest(self):
        for key in ("work_sha256", "hidden_sha256", "verification_evidence_sha256"):
            self.assertIn(key, cp.DETERMINISTIC)


class TestTreeAndEvidence(Quiet):
    def tree(self, files: dict[str, str]) -> str:
        root = tempfile.mkdtemp(dir=self.tmp)
        for rel, text in files.items():
            path = pathlib.Path(root, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root

    def test_content_name_and_extra_files_all_change_the_digest(self):
        base = cp.tree_digest(self.tree({"a.py": "x", "d/b.py": "y"}))
        for other in (
            {"a.py": "X", "d/b.py": "y"},
            {"a2.py": "x", "d/b.py": "y"},
            {"a.py": "x", "d/b.py": "y", "c.py": ""},
            {"a.py": "x"},
        ):
            self.assertNotEqual(cp.tree_digest(self.tree(other)), base, other)
        self.assertEqual(cp.tree_digest(self.tree({"a.py": "x", "d/b.py": "y"})), base)

    def test_caches_and_git_are_ignored_but_symlinks_are_not(self):
        base = self.tree({"a.py": "x"})
        digest = cp.tree_digest(base)
        for noise in (".git/HEAD", "__pycache__/a.pyc", ".pytest_cache/v"):
            pathlib.Path(base, noise).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(base, noise).write_text("noise")
        self.assertEqual(cp.tree_digest(base), digest)
        os.symlink("a.py", os.path.join(base, "link"))
        self.assertNotEqual(cp.tree_digest(base), digest)

    def test_a_symlink_is_not_the_same_as_a_copy_of_its_target(self):
        copy = self.tree({"a.py": "x", "b.py": "x"})
        link = self.tree({"a.py": "x"})
        os.symlink("a.py", os.path.join(link, "b.py"))
        self.assertNotEqual(cp.tree_digest(copy), cp.tree_digest(link))

    def test_evidence_ignores_timing_but_not_the_result(self):
        fast = "Ran 3 tests in 0.001s\n\nOK"
        slow = "Ran 3 tests in 1.250s\n\nOK"
        self.assertEqual(cp.evidence_digest(fast), cp.evidence_digest(slow))
        for other in ("Ran 2 tests in 0.001s\n\nOK", "Ran 3 tests in 0.001s\n\nFAILED"):
            self.assertNotEqual(cp.evidence_digest(other), cp.evidence_digest(fast))
        self.assertNotEqual(cp.evidence_digest(""), cp.evidence_digest(fast))


class TestReference(Quiet):
    def test_the_reference_is_built_without_the_engine_and_passes(self):
        ref = cp.reference(TASK, self.tmp)
        self.assertEqual(ref["tests_exit"], 0)
        self.assertEqual(ref["task_digest"], engine.load_suite("v0.5")["tasks"][TASK])
        for key in ("work_sha256", "hidden_sha256", "verification_evidence_sha256"):
            self.assertRegex(ref[key], r"^[0-9a-f]{64}$")
        self.assertEqual(cp.reference_problems(TASK, ref), [])

    def test_every_calibration_task_passes_its_own_independent_tests(self):
        refs, problems = cp.build_references(cp.workload()["calibration"])
        self.assertEqual(problems, [])
        self.assertEqual(len(refs), 14)

    def test_a_solution_that_fails_its_tests_is_reported(self):
        ref = {**cp.reference(TASK, self.tmp), "tests_exit": 1}
        self.assertTrue(cp.reference_problems(TASK, ref))
        self.assertTrue(cp.reference_problems(TASK, {**ref, "tests_exit": None}))


class TestIntegrity(unittest.TestCase):
    REF: ClassVar[dict[str, Any]] = {
        "work_sha256": "w",
        "hidden_sha256": "h",
        "verification_evidence_sha256": "e",
        "task_digest": "d",
    }
    EVIDENCE: ClassVar[dict[str, Any]] = {
        "work_sha256": "w",
        "hidden_sha256": "h",
        "verification_evidence_sha256": "e",
    }

    def problems(self, row=None, evidence=None, ref="default"):
        return cp.integrity_problems(
            good_row() if row is None else row,
            dict(self.EVIDENCE) if evidence is None else evidence,
            dict(self.REF) if ref == "default" else ref,
        )

    def test_a_matching_attempt_has_no_problem(self):
        self.assertEqual(self.problems(), [])

    def test_each_disk_measurement_must_match_the_independent_reference(self):
        for key in self.EVIDENCE:
            found = self.problems(evidence={**self.EVIDENCE, key: "other"})
            self.assertTrue(any(key in p for p in found), (key, found))

    def test_skipped_verification_leaves_no_evidence_and_fails(self):
        key = "verification_evidence_sha256"
        row = {k: v for k, v in good_row().items() if k != key}
        evidence = {k: v for k, v in self.EVIDENCE.items() if k != key}
        found = self.problems(row=row, evidence=evidence)
        self.assertIn(f"missing field {key}", found)

    def test_other_failures(self):
        self.assertTrue(self.problems(row={}))
        self.assertEqual(
            cp.integrity_problems(None, {}, self.REF), ["no readable results row"]
        )
        self.assertTrue(self.problems(row={**good_row(), "passed": False}))
        self.assertTrue(self.problems(ref=None))
        moved = self.problems(row={**good_row(), "task_digest": "x"})
        self.assertTrue(any("suite pin" in p for p in moved), moved)


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

    def setUp(self) -> None:
        super().setUp()
        self.refs, problems = cp.build_references([TASK])
        self.assertEqual(problems, [])
        self.failed = os.path.join(self.tmp, "failed")

    def p(self, **kw: Any) -> cp.Plan:
        return plan(tasks=[TASK], refs=self.refs, failed_dir=self.failed, **kw)

    def test_a_scripted_attempt_passes_matches_the_reference_and_reproduces(self):
        first = cp.record(self.p(), TASK, 0, self.tmp)
        second = cp.record(self.p(), TASK, 1, self.tmp)
        for row in (first, second):
            self.assertTrue(row["ok"], row)
            self.assertEqual(row["problems"], [])
            self.assertEqual(row["exit"], 0)
            self.assertGreater(row["wall_s"], 0)
            self.assertGreater(row["maxrss_kib"], 0)
            self.assertAlmostEqual(row["cpu_s"], row["user_s"] + row["sys_s"])
            self.assertEqual((row["commands_executed"], row["model_turns"]), (1, 2))
            self.assertEqual(row["work_sha256"], self.refs[TASK]["work_sha256"])
        self.assertEqual(first["digest"], second["digest"])
        self.assertTrue(first["digest"])
        self.assertFalse(os.path.exists(self.failed))  # nothing failed, nothing kept

    def test_an_actor_that_does_not_solve_the_task_is_a_failure(self):
        run_dir = os.path.join(self.tmp, "idle")
        with (
            mock.patch.object(cp, "solution_command", lambda d: "true"),
            mock.patch.dict(os.environ, {"BENCH_MOCK_MODEL": "1"}),
        ):
            cp.one_main(["--task", TASK, "--run-dir", run_dir, "--no-strict-isolation"])
        row = cp.read_row(run_dir)
        assert row is not None
        self.assertFalse(row["passed"])
        found = cp.integrity_problems(
            row, cp.disk_evidence(run_dir, TASK), self.refs[TASK]
        )
        self.assertTrue(any("work_sha256 differs" in p for p in found), found)
        self.assertTrue(any("not report the task solved" in p for p in found))

    def test_a_nonzero_child_exit_fails_even_when_the_row_says_passed(self):
        real = cp.measure_child

        def child_exits_1(*args: Any, **kwargs: Any):
            outer, _code, timed_out = real(*args, **kwargs)
            return outer, 1, timed_out

        with mock.patch.object(cp, "measure_child", child_exits_1):
            row = cp.record(self.p(), TASK, 0, self.tmp)
        self.assertFalse(row["ok"])
        self.assertEqual(row["problems"], ["the child exited with status 1"])

    def test_a_failed_attempt_keeps_its_directory_and_log(self):
        real = cp.measure_child
        with mock.patch.object(
            cp, "measure_child", lambda *a, **k: (real(*a, **k)[0], 2, False)
        ):
            cp.record(self.p(), TASK, 0, self.tmp)
        kept = os.path.join(self.failed, f"calibration-{TASK}-0-1")
        self.assertTrue(os.path.exists(os.path.join(kept, "child.log")))
        self.assertTrue(os.path.exists(os.path.join(kept, "results.json")))
        self.assertTrue(os.path.isdir(os.path.join(kept, "r0", TASK, "work")))

    def test_a_hung_child_is_killed_recorded_and_kept(self):
        row = cp.record(self.p(child_timeout_s=0.05), TASK, 0, self.tmp)
        self.assertTrue(row["timed_out"])
        self.assertFalse(row["ok"])
        self.assertTrue(any("killed after" in p for p in row["problems"]))
        self.assertTrue(os.path.isdir(self.failed))

    def test_a_missing_deterministic_field_makes_the_run_fail(self):
        real = cp.read_row

        def without_tokens(run_dir: str):
            row = real(run_dir)
            assert row is not None
            row.pop("tokens")
            return row

        with mock.patch.object(cp, "read_row", without_tokens):
            row = cp.record(self.p(), TASK, 0, self.tmp)
        self.assertFalse(row["ok"])
        self.assertIsNone(row["digest"])
        self.assertIn("missing field tokens", row["problems"])

    def test_a_workspace_that_differs_from_the_reference_is_a_failure(self):
        wrong = {**self.refs, TASK: {**self.refs[TASK], "work_sha256": "0" * 64}}
        row = cp.record(plan(tasks=[TASK], refs=wrong), TASK, 0, self.tmp)
        self.assertFalse(row["ok"])
        self.assertTrue(any("work_sha256 differs" in p for p in row["problems"]))

    def test_a_run_without_a_reference_is_a_failure(self):
        row = cp.record(plan(tasks=[TASK], refs={}), TASK, 0, self.tmp)
        self.assertFalse(row["ok"])


class TestBatches(Quiet):
    def test_workers_are_capped(self):
        with mock.patch.object(cp, "record", fake_record()):
            rows, batch = cp.run_batches(plan(workers=99))
        self.assertEqual(batch["workers"], cp.MAX_WORKERS)
        self.assertEqual(len(rows), 2)

    def test_the_cpu_budget_stops_admission_and_says_so(self):
        with mock.patch.object(cp, "record", fake_record(cpu=10.0)):
            rows, batch = cp.run_batches(plan(tasks=list("abcdef"), budget_s=25.0))
        self.assertEqual(len(rows), 2)  # 20 spent + 10 for the next >= 25
        self.assertEqual((batch["attempted"], batch["planned"]), (2, 6))
        self.assertIn("budget", batch["stopped"])

    def test_the_worst_run_so_far_times_the_wave_is_reserved(self):
        # two workers, 10 s each: after one wave 20 are spent; another wave could
        # cost 20 more, which would pass the 35 s budget, so it is not admitted
        with mock.patch.object(cp, "record", fake_record(cpu=10.0)):
            rows, batch = cp.run_batches(
                plan(tasks=list("abcd"), workers=2, budget_s=35.0)
            )
        self.assertEqual(len(rows), 2)
        self.assertTrue(batch["stopped"])

    def test_cpu_already_spent_counts_against_the_budget(self):
        with mock.patch.object(cp, "record", fake_record(cpu=1.0)):
            rows, batch = cp.run_batches(plan(tasks=["a"], budget_s=20.0, spent_s=20.0))
        self.assertEqual(rows, [])
        self.assertIsNotNone(batch["stopped"])

    def test_summary_reports_failures_and_nondeterminism(self):
        a = fake_record()(plan(), "t", 0, "")
        b = {**a, "rep": 1, "digest": "other"}
        c = {**a, "task": "u", "ok": False, "digest": None}
        group = cp.summary([a, b, c], [])["groups"]["calibration/workers=1"]
        self.assertEqual(group["execution_failures"], 1)
        self.assertEqual(group["nondeterministic_tasks"], ["t", "u"])
        self.assertEqual(cp.digests([a, b, c]), {})

    def test_digests_are_taken_over_every_worker_count(self):
        a = fake_record()(plan(workers=1), "t", 0, "")
        other = fake_record(digest=lambda p, t: "different")
        b = other(plan(workers=2), "t", 0, "")
        self.assertEqual(cp.digests([a]), {"t": "d"})
        self.assertEqual(cp.digests([a, b]), {})  # same task, two digests


class TestValidity(unittest.TestCase):
    @staticmethod
    def batch(**kw: Any) -> dict[str, Any]:
        base = {"set": "calibration", "workers": 1, "attempted": 2, "planned": 2,
                "stopped": None, "wall_s": 1.0, "workflows_per_hour": 1.0}  # fmt: skip
        return {**base, **kw}

    @staticmethod
    def rows(**kw: Any) -> list[dict[str, Any]]:
        rec = fake_record(**kw)
        return [rec(plan(), t, 0, "") for t in ("a", "b")]

    @staticmethod
    def check(rows, batches, budget=100.0, refs=()):
        return cp.validity_problems(
            rows, batches, budget_s=budget, reference_problem_list=list(refs)
        )

    def test_a_complete_consistent_matrix_is_valid(self):
        self.assertEqual(self.check(self.rows(), [self.batch()]), [])

    def test_an_incomplete_matrix_is_invalid(self):
        found = self.check(self.rows()[:1], [self.batch(attempted=1, stopped="cpu")])
        self.assertTrue(any("incomplete matrix" in p for p in found), found)
        # a short batch is incomplete even without a recorded stop reason
        found = self.check(self.rows()[:1], [self.batch(attempted=1)])
        self.assertTrue(any("incomplete matrix" in p for p in found), found)

    def test_a_failed_run_is_invalid(self):
        found = self.check(self.rows(ok=False), [self.batch()])
        self.assertTrue(any("boom" in p for p in found), found)

    def test_nondeterminism_within_a_group_is_invalid(self):
        rows = self.rows()
        rows.append({**rows[0], "rep": 1, "digest": "other"})
        found = self.check(rows, [self.batch(attempted=3, planned=3)])
        self.assertTrue(any("no single deterministic digest" in p for p in found))

    def test_different_digests_across_worker_counts_are_invalid(self):
        one = self.rows()
        two = [{**r, "workers": 2, "digest": "other"} for r in self.rows()]
        found = self.check([*one, *two], [self.batch(), self.batch(workers=2)])
        self.assertTrue(any("no single deterministic digest" in p for p in found))

    def test_a_missing_digest_is_invalid(self):
        found = self.check(self.rows(digest=lambda p, t: None), [self.batch()])
        self.assertTrue(any("no single deterministic digest" in p for p in found))

    def test_overshooting_the_cpu_budget_is_reported(self):
        found = self.check(self.rows(cpu=30.0), [self.batch()], budget=50.0)
        self.assertTrue(any("budget exceeded" in p for p in found), found)

    def test_reference_problems_are_carried(self):
        found = self.check(self.rows(), [self.batch()], refs=["x did not pass"])
        self.assertIn("x did not pass", found)


class TestCli(Quiet):
    def run_cli(
        self, *argv: str, record: Any = None, refs: tuple[dict, list] = ({}, [])
    ) -> int:
        doc = {"calibration": ["a", "b"], "final": ["c"]}
        with (
            mock.patch.object(cp, "workload", lambda: doc),
            mock.patch.object(cp, "build_references", lambda tasks: refs),
            mock.patch.object(cp, "record", record or fake_record()),
        ):
            return cp.main(["run", *argv])

    def out(self) -> str:
        return os.path.join(self.tmp, "o")

    def summary(self) -> dict[str, Any]:
        with open(os.path.join(self.out(), "summary.json")) as f:
            return json.load(f)

    def test_the_final_set_needs_an_explicit_flag(self):
        self.assertEqual(self.run_cli("--set", "final", "--out", self.out()), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out(), "measurements.jsonl")))

    def test_an_existing_result_is_never_overwritten(self):
        os.makedirs(self.out())
        with open(os.path.join(self.out(), "summary.json"), "w") as f:
            json.dump({"keep": True}, f)
        self.assertEqual(self.run_cli("--set", "calibration", "--out", self.out()), 2)
        self.assertEqual(self.summary(), {"keep": True})

    def test_a_complete_valid_run_exits_zero_and_writes_everything(self):
        code = self.run_cli(
            "--set", "calibration", "--reps", "2", "--workers", "1", "2",
            "--out", self.out(), "--no-strict-isolation",
        )  # fmt: skip
        self.assertEqual(code, 0)
        with open(os.path.join(self.out(), "measurements.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 8)
        doc = self.summary()
        self.assertTrue(doc["valid"])
        self.assertEqual(doc["problems"], [])
        self.assertEqual(
            sorted(doc["groups"]), ["calibration/workers=1", "calibration/workers=2"]
        )
        self.assertIn("workload_sha256", doc)
        self.assertIn("admission-control", doc["cpu_budget_mode"])
        self.assertEqual(cp.main(["report", self.out()]), 0)

    def invalid(self, *argv: str, record: Any = None) -> dict[str, Any]:
        code = self.run_cli(
            "--set", "calibration", "--out", self.out(), *argv, record=record
        )
        self.assertEqual(code, 1, self.summary()["problems"])
        doc = self.summary()
        self.assertFalse(doc["valid"])
        self.assertEqual(cp.main(["report", self.out()]), 1)
        return doc

    def test_a_failed_attempt_makes_the_run_fail(self):
        self.invalid(record=fake_record(ok=False))

    def test_nondeterministic_outputs_make_the_run_fail_even_if_all_passed(self):
        counter = iter(range(100))
        doc = self.invalid(
            "--reps", "2", record=fake_record(digest=lambda p, t: f"d{next(counter)}")
        )
        self.assertTrue(any("deterministic digest" in p for p in doc["problems"]))

    def test_digests_that_differ_across_worker_counts_make_the_run_fail(self):
        doc = self.invalid(
            "--workers",
            "1",
            "2",
            record=fake_record(digest=lambda p, t: f"w{p.workers}"),
        )
        self.assertTrue(any("deterministic digest" in p for p in doc["problems"]))

    def test_a_partial_matrix_from_the_budget_exits_nonzero_and_keeps_evidence(self):
        doc = self.invalid(
            "--reps", "3", "--cpu-budget", "25", record=fake_record(cpu=10.0)
        )
        self.assertTrue(any("incomplete matrix" in p for p in doc["problems"]))
        with open(os.path.join(self.out(), "measurements.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 2)  # what ran is kept

    def test_a_worker_count_that_was_not_run_makes_the_run_fail(self):
        # the budget stops the first batch, so the second worker count never runs
        doc = self.invalid(
            "--workers", "1", "2", "--cpu-budget", "1", record=fake_record(cpu=0.9)
        )
        self.assertTrue(any("incomplete matrix" in p for p in doc["problems"]))
        self.assertEqual(len(doc["batches"]), 1)

    def test_an_unsolvable_reference_stops_the_run_before_any_attempt(self):
        calls: list[str] = []

        def record(plan, task, rep, scratch):
            calls.append(task)
            return fake_record()(plan, task, rep, scratch)

        code = self.run_cli(
            "--set", "calibration", "--out", self.out(),
            record=record, refs=({}, ["a: tests fail"]),
        )  # fmt: skip
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("a: tests fail", self.summary()["problems"])

    def test_the_cpu_budget_argument_is_bounded(self):
        for bad in ("0", "-1", "nan", "inf", "1201", "text"):
            with self.assertRaises(SystemExit, msg=bad):
                self.run_cli("--set", "calibration", "--cpu-budget", bad)
        self.assertEqual(cp.budget_arg("1200"), 1200.0)
        self.assertEqual(cp.budget_arg("0.5"), 0.5)


if __name__ == "__main__":
    unittest.main()
