"""Offline tests for the timing-breakdown tool (no GPU, no model)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from typing import Any
from unittest import mock

from experiments import cpu_breakdown as bd
from experiments import cpu_profile as cp

TASK = "debugging-off-by-one"


def sp(name: str, t0: float, t1: float, thread: str = bd.MAIN) -> dict[str, Any]:
    return {
        "kind": "span",
        "name": name,
        "thread": thread,
        "t0": t0,
        "t1": t1,
        "cpu_s": 0.0,
    }


class TestExclusiveTimes(unittest.TestCase):
    def test_nested_spans_are_not_double_counted(self):
        spans = [
            sp("main", 0, 10),
            sp("sandbox", 2, 6),
            sp("probe", 2, 3),
            sp("kill", 5, 6),
        ]
        got = bd.exclusive_times(spans)
        self.assertEqual(got, {"main": 6, "sandbox": 2, "probe": 1, "kill": 1})
        self.assertAlmostEqual(sum(got.values()), 10)  # nothing lost or counted twice

    def test_siblings_and_repeats_add_up(self):
        spans = [sp("main", 0, 10), sp("sandbox", 1, 3), sp("sandbox", 4, 7)]
        got = bd.exclusive_times(spans)
        self.assertEqual(got["sandbox"], 5)
        self.assertEqual(got["main"], 5)


class TestBreakdown(unittest.TestCase):
    def trace(self) -> dict[str, Any]:
        proc = {
            "kind": "proc",
            "tag": "git",
            "thread": bd.MAIN,
            "t0": 1.0,
            "t1": 1.4,
            "spawn_s": 0.01,
        }
        events = [
            sp("engine_main", 0, 10.5),
            sp("sandbox", 2, 6),
            sp("cgroup_probe", 2, 3),
            sp("monitor_stop", 8, 9),
            sp("monitor_sample", 1, 2, "Thread-1"),
            proc,
        ]
        return {
            "spawn": 100.0,
            "t_first": 100.5,
            "t_entry": 101.5,
            "t_exit": 112.0,
            "pynvml": False,
            "events": events,
        }

    def test_components_add_up_to_the_measured_wall(self):
        out = bd.breakdown(self.trace(), {"wall_s": 12.5})
        comp = out["components"]
        self.assertAlmostEqual(comp["startup"], 0.5)
        self.assertAlmostEqual(comp["imports"], 1.0)
        self.assertAlmostEqual(comp["teardown"], 0.5)  # 12.5 - (112 - 100)
        self.assertAlmostEqual(comp["cgroup_probe"], 1.0)
        self.assertAlmostEqual(comp["sandbox_run"], 3.0)
        self.assertAlmostEqual(comp["monitor_stop"], 1.0)
        self.assertAlmostEqual(out["critical_path_s"], 12.5)

    def test_background_monitor_samples_are_not_on_the_critical_path(self):
        out = bd.breakdown(self.trace(), {"wall_s": 12.5})
        self.assertEqual(out["monitor"]["samples"], 1)
        self.assertEqual(out["monitor"]["backend"], "nvidia-smi")
        self.assertNotIn("monitor_sample", out["components"])
        self.assertAlmostEqual(out["procs"]["git"]["life_s"], 0.4)


class TestInstrumentation(unittest.TestCase):
    def setUp(self) -> None:
        saved = list(bd.EVENTS)
        bd.EVENTS.clear()
        self.addCleanup(lambda: (bd.EVENTS.clear(), bd.EVENTS.extend(saved)))

    def test_span_records_time_and_passes_the_result_through(self):
        def slow(x: int) -> int:
            time.sleep(0.02)
            return x + 1

        self.assertEqual(bd.span("s", slow)(1), 2)
        (ev,) = bd.EVENTS
        self.assertEqual(ev["name"], "s")
        self.assertGreaterEqual(ev["t1"] - ev["t0"], 0.02)

    def test_a_raising_function_is_still_recorded_and_still_raises(self):
        def boom() -> None:
            raise ValueError("x")

        with self.assertRaises(ValueError):
            bd.span("b", boom)()
        self.assertEqual(bd.EVENTS[0]["name"], "b")

    def test_popen_records_spawn_and_lifetime_and_tags_the_scope_launcher(self):
        with mock.patch.object(bd.subprocess, "Popen", bd.TimedPopen):
            bd.subprocess.run(["true"], check=True)
        (ev,) = bd.EVENTS
        self.assertEqual(ev["tag"], "true")
        self.assertIn("t1", ev)
        self.assertGreaterEqual(ev["spawn_s"], 0)
        self.assertEqual(
            bd._tag(["systemd-run", "--scope", "true"]), "systemd-run:probe"
        )
        self.assertEqual(
            bd._tag(["systemd-run", "--scope", "--", "bwrap"]), "systemd-run:scope"
        )

    def test_install_patches_only_this_process(self):
        mon = bd.telemetry.VramMonitor
        saved = (subprocess.Popen, mon.start, mon.stop, bd.telemetry.get_gpu_memory)
        saved_engine = {a: getattr(bd.engine, a) for a in bd.SPANS}
        self.addCleanup(
            lambda: [setattr(bd.engine, a, f) for a, f in saved_engine.items()]
        )
        self.addCleanup(
            lambda: (
                setattr(subprocess, "Popen", saved[0]),
                setattr(mon, "start", saved[1]),
                setattr(mon, "stop", saved[2]),
                setattr(bd.telemetry, "get_gpu_memory", saved[3]),
            )
        )
        bd.install()
        self.assertIs(subprocess.Popen, bd.TimedPopen)
        self.assertIsNot(bd.engine.run_sandboxed, saved_engine["run_sandboxed"])


class TestAnalysis(unittest.TestCase):
    @staticmethod
    def run_(
        kind: str, wall: float, pressure: float, monitor_stop: float = 0.5
    ) -> dict[str, Any]:
        comp = {c: 0.0 for c in bd.COMPONENTS} | {
            "monitor_stop": monitor_stop,
            "engine_other": 0.5,
        }
        row: dict[str, Any] = {
            "kind": kind,
            "wall_s": wall,
            "cpu_s": 0.4,
            "cpu_pressure10": pressure,
        }
        if kind == "traced":
            row["breakdown"] = {
                "components": comp,
                "critical_path_s": sum(comp.values()),
                "sandbox_calls": 3,
                "per_call": {
                    "cgroup_probe": 0.02,
                    "sandbox_run": 0.07,
                    "cgroup_kill": 0.006,
                },
                "procs": {},
                "monitor": {
                    "backend": "nvidia-smi",
                    "samples": 1,
                    "wall_s": 0.06,
                    "cpu_s": 0.0,
                },
            }
        return row

    def test_overhead_and_inflation_flag(self):
        rows = [self.run_("traced", 1.5, p) for p in (1, 2, 3, 4)]
        rows += [self.run_("plain", 1.0, p) for p in (1, 2, 3, 4)]
        out = bd.analyse(rows)
        self.assertAlmostEqual(out["overhead"]["wall_s"], 0.5)
        self.assertTrue(out["inflated"])  # more than 10% of plain wall
        near = [self.run_("traced", 1.05, 1), self.run_("plain", 1.0, 1)]
        self.assertFalse(bd.analyse(near)["inflated"])

    def test_shares_sum_to_one_and_load_halves_are_split_at_the_median(self):
        rows = [self.run_("traced", 1.0, p) for p in (1, 2, 3, 4)] + [
            self.run_("plain", 1.0, 1)
        ]
        out = bd.analyse(rows)
        self.assertAlmostEqual(
            sum(c["share"] for c in out["all"]["components"].values()), 1.0
        )
        self.assertEqual(out["low_load"]["runs"], 2)
        self.assertEqual(out["high_load"]["runs"], 2)
        self.assertTrue(out["by_load_ranking_agrees"])

    def test_disagreeing_load_halves_are_reported_as_such(self):
        rows = [
            self.run_("traced", 1.0, 1, monitor_stop=0.9),
            self.run_("traced", 1.0, 1, monitor_stop=0.9),
            self.run_("traced", 1.0, 9, monitor_stop=0.0),
            self.run_("traced", 1.0, 9, monitor_stop=0.0),
            self.run_("plain", 1.0, 1),
        ]
        self.assertFalse(bd.analyse(rows)["by_load_ranking_agrees"])

    def test_nothing_to_compare_without_both_kinds(self):
        self.assertNotIn("overhead", bd.analyse([self.run_("traced", 1.0, 1)]))


class TestRealAttempt(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.tmp = tempfile.mkdtemp(prefix="cpu-breakdown-test-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_a_traced_attempt_matches_an_untraced_one_and_accounts_for_its_wall(self):
        refs = {TASK: cp.reference(TASK, self.tmp)}
        traced = bd.attempt(TASK, "traced", refs, self.tmp, strict=False)
        plain = bd.attempt(TASK, "plain", refs, self.tmp, strict=False)
        self.assertTrue(traced["ok"], traced["problems"])
        self.assertTrue(plain["ok"], plain["problems"])
        self.assertEqual(traced["digest"], plain["digest"])  # tracing changes no output
        crit = traced["breakdown"]["critical_path_s"]
        self.assertAlmostEqual(crit, traced["wall_s"], delta=0.1 * traced["wall_s"])
        self.assertGreaterEqual(traced["breakdown"]["sandbox_calls"], 1)

    def test_a_failing_attempt_is_not_ok(self):
        refs = {TASK: {**cp.reference(TASK, self.tmp), "work_sha256": "0" * 64}}
        got = bd.attempt(TASK, "plain", refs, self.tmp, strict=False)
        self.assertFalse(got["ok"])
        self.assertTrue(any("work_sha256" in p for p in got["problems"]))


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.mkdtemp(prefix="cpu-breakdown-test-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_reps_and_budget_are_validated(self):
        for bad in (["--reps", "0"], ["--cpu-budget", "0"], ["--cpu-budget", "5000"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                bd.main(["run", "--out", os.path.join(self.tmp, "o"), *bad])

    def test_an_existing_result_is_never_overwritten(self):
        out = os.path.join(self.tmp, "o")
        os.makedirs(out)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"keep": True}, f)
        self.assertEqual(bd.main(["run", "--out", out]), 2)

    def test_a_failed_attempt_makes_the_run_invalid_and_exit_nonzero(self):
        def bad(task: str, kind: str, *a: Any, **k: Any) -> dict[str, Any]:
            return {
                "task": task,
                "kind": kind,
                "ok": False,
                "problems": ["boom"],
                "cpu_s": 0.1,
                "digest": None,
                "wall_s": 1.0,
            }

        out = os.path.join(self.tmp, "o")
        with (
            mock.patch.object(bd.cp, "build_references", lambda tasks: ({}, [])),
            mock.patch.object(bd, "attempt", bad),
        ):
            code = bd.main(["run", "--reps", "1", "--out", out])
        self.assertEqual(code, 1)
        with open(os.path.join(out, "summary.json")) as f:
            self.assertFalse(json.load(f)["valid"])

    def test_the_cpu_budget_stops_the_run_and_marks_it_invalid(self):
        def ok(task: str, kind: str, *a: Any, **k: Any) -> dict[str, Any]:
            return {
                "task": task,
                "kind": kind,
                "ok": True,
                "problems": [],
                "cpu_s": 10.0,
                "digest": "d",
                "wall_s": 1.0,
            }

        out = os.path.join(self.tmp, "o")
        with (
            mock.patch.object(bd.cp, "build_references", lambda tasks: ({}, [])),
            mock.patch.object(bd, "attempt", ok),
        ):
            code = bd.main(["run", "--reps", "1", "--cpu-budget", "25", "--out", out])
        self.assertEqual(code, 1)
        with open(os.path.join(out, "summary.json")) as f:
            self.assertTrue(any("budget" in p for p in json.load(f)["problems"]))


if __name__ == "__main__":
    unittest.main()
