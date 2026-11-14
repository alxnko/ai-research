"""Offline tests of the experiment controller (no Ollama, GPU or network).

The reference data is `experiments/exp2_records/`: byte-for-byte copies of the
36 EXP-2 `config.json`/`exp2_policy.json`/`results.json` files (hashes in
SHA256SUMS). Every defect test runs on a throw-away copy, with a clean twin.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
import unittest
from typing import Any

from experiments import analyze_exp2, controller

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifests", "exp2.json")
RECORDS = os.path.join(HERE, "exp2_records")
Key = tuple[str, int, str]
CELL: Key = ("control", 42, "backend-auth-token")


def manifest() -> dict[str, Any]:
    return controller.load_manifest(MANIFEST)


def edit(path: str, change) -> None:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc = change(doc) or doc
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


class Scratch(unittest.TestCase):
    """A private copy of the EXP-2 records for each test."""

    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.mkdtemp(prefix="controller-test-", dir="/var/tmp")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = os.path.join(self.tmp, "runs")
        shutil.copytree(RECORDS, self.runs)

    def cell(self, key: Key = CELL) -> str:
        return controller.cell_dir(self.runs, *key)


class TestReproduction(Scratch):
    def test_exp2_numbers_from_the_original_records(self):
        report = controller.analyze(manifest(), RECORDS)
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"], {"valid": 36, "missing": 0, "invalid": 0})
        control, verification = (
            report["arms"]["control"],
            report["arms"]["verification"],
        )
        self.assertEqual((control["successes"], control["attempts"]), (6, 18))
        self.assertEqual((verification["successes"], verification["attempts"]), (9, 18))
        self.assertEqual(report["decision"]["ratio"], 1.51)
        self.assertTrue(report["decision"]["met"])
        self.assertEqual(len(report["historical_aborted"]), 2)

    def test_agrees_with_the_frozen_analyzer(self):
        report = controller.analyze(manifest(), RECORDS)
        original_runs = analyze_exp2.RUNS
        analyze_exp2.RUNS = RECORDS
        try:
            frozen = analyze_exp2.load()
        finally:
            analyze_exp2.RUNS = original_runs
        for arm in analyze_exp2.ARMS:
            old = analyze_exp2.summarize(frozen, arm)
            new = report["arms"][arm]
            for key in (
                "attempts",
                "successes",
                "total_wall_s",
                "verified_tasks_per_hour",
                "median_wall_s",
                "p95_wall_s",
                "tokens_evaluated",
                "inference_errors",
                "timeouts",
                "invalid_actions",
                "peak_vram_mib",
            ):
                self.assertEqual(new[key], old[key], f"{arm}.{key}")
            self.assertEqual(new["wilson95"], old["wilson95"])

    def test_copies_match_their_recorded_hashes(self):
        with open(os.path.join(RECORDS, "SHA256SUMS"), encoding="utf-8") as f:
            entries = [line.split("  ", 1) for line in f.read().splitlines()]
        self.assertEqual(len(entries), 108)
        for digest, rel in entries:
            with open(os.path.join(RECORDS, rel), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, rel)

    def test_copies_match_the_original_run_records_when_present(self):
        originals = os.environ.get("EXP2_RUNS") or os.path.join(
            controller.ROOT, "runs", "exp2"
        )
        if not os.path.isdir(os.path.join(originals, "control")):
            self.skipTest("original runs/exp2 not present (SKIPPED, not verified)")
        with open(os.path.join(RECORDS, "SHA256SUMS"), encoding="utf-8") as f:
            for line in f.read().splitlines():
                digest, rel = line.split("  ", 1)
                with open(os.path.join(originals, rel), "rb") as g:
                    self.assertEqual(hashlib.sha256(g.read()).hexdigest(), digest, rel)

    def test_attempt_identities(self):
        m = manifest()
        ids = {controller.attempt_id(m, *key) for key in controller.attempts(m)}
        self.assertEqual(len(ids), 36)
        self.assertEqual(
            controller.attempt_id(m, *CELL), controller.attempt_id(m, *CELL)
        )
        changed = copy.deepcopy(m)
        changed["settings"]["steps"] = 21
        self.assertNotEqual(
            controller.attempt_id(m, *CELL), controller.attempt_id(changed, *CELL)
        )
        order = list(controller.attempts(m))
        self.assertEqual(order[0], CELL)
        self.assertEqual(order[18][0], "verification")


class TestManifestSchema(unittest.TestCase):
    def test_clean_manifest_is_valid(self):
        self.assertEqual(controller.validate_manifest(manifest()), [])

    def test_each_defect_is_rejected(self):
        def drop(key):
            return lambda m: m.pop(key)

        def setpath(path, value):
            def change(m):
                node = m
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = value

            return change

        defects = {
            "missing key": drop("model"),
            "unknown key": setpath(["surprise"], 1),
            "wrong type": setpath(["settings", "ctx"], "16384"),
            "bool as int": setpath(["settings", "steps"], True),
            "duplicate seeds": setpath(["seeds"], [42, 42]),
            "empty tasks": setpath(["tasks"], []),
            "bad prompt hash": setpath(["arms", "control", "prompt_sha256"], "abc"),
            "unknown arm key": setpath(["arms", "control", "extra"], 1),
            "analysis names no arm": setpath(["analysis", "control"], "nope"),
            "wrong schema": setpath(["schema"], 2),
            "missing digest": setpath(["model", "weights_digest"], None),
        }
        for label, change in defects.items():
            m = copy.deepcopy(manifest())
            change(m)
            self.assertTrue(controller.validate_manifest(m), label)


class TestProvenance(Scratch):
    """Each tampered record is rejected; the untouched twin is valid."""

    def problems(self, key: Key = CELL) -> list[str]:
        return controller.validate_attempt(manifest(), *key, self.cell(key))

    def test_clean_attempt_is_valid(self):
        self.assertEqual(self.problems(), [])

    def test_each_defect_is_detected(self):
        policy, results, config = "exp2_policy.json", "results.json", "config.json"
        defects = {
            "other arm's prompt": (policy, lambda d: d.update(prompt_sha256="0" * 64)),
            "wrong model weights": (
                policy,
                lambda d: d["model_fingerprint"].update(weights_digest="sha256:bad"),
            ),
            "other benchmark tag": (policy, lambda d: d.update(benchmark_tag="v0.5")),
            "altered invocation": (
                policy,
                lambda d: d["benchmark_argv"].__setitem__(
                    d["benchmark_argv"].index("--steps") + 1, "99"
                ),
            ),
            "wrong policy": (policy, lambda d: d.update(policy="verification")),
            "fixture digest drift": (
                results,
                lambda d: d[0].update(task_digest="0" * 64),
            ),
            "other task": (results, lambda d: d[0].update(task="refactor-parser")),
            "two records": (results, lambda d: d.append(dict(d[0]))),
            "non-boolean verdict": (results, lambda d: d[0].update(run_successful=1)),
            "isolation off": (results, lambda d: d[0].update(strict_isolation=False)),
            "negative wall time": (results, lambda d: d[0].update(total_wall_s=-1)),
            "sampling drift": (
                results,
                lambda d: d[0]["sampling"].update(temperature=0.7),
            ),
            "other seed": (config, lambda d: d["sampling"].update(seed=43)),
            "other model tag": (config, lambda d: d.update(model="qwen3.5:4b")),
            "other budget": (config, lambda d: d.update(max_wall=1200)),
        }
        for label, (name, change) in defects.items():
            scratch = os.path.join(self.tmp, "twin")
            shutil.rmtree(scratch, ignore_errors=True)
            shutil.copytree(self.cell(), scratch)
            edit(os.path.join(scratch, name), change)
            found = controller.validate_attempt(manifest(), *CELL, scratch)
            self.assertTrue(found, f"{label} was accepted")

    def test_missing_or_unreadable_files_are_detected(self):
        os.remove(os.path.join(self.cell(), "exp2_policy.json"))
        self.assertTrue(self.problems())
        with open(os.path.join(self.cell(), "results.json"), "w") as f:
            f.write("{not json")
        self.assertTrue(any("results.json" in p for p in self.problems()))


class TestMatrixEnforcement(Scratch):
    def test_complete_matrix_is_enforced(self):
        shutil.rmtree(self.cell(("verification", 44, "refactor-validators")))
        report = controller.analyze(manifest(), self.runs)
        self.assertFalse(report["complete"])
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["decision"]["status"], "withheld")
        self.assertEqual(controller.main(["analyze", MANIFEST, "--runs", self.runs]), 2)

    def test_invalid_attempt_blocks_the_decision(self):
        edit(
            os.path.join(self.cell(), "exp2_policy.json"),
            lambda d: d.update(prompt_sha256="0" * 64),
        )
        report = controller.analyze(manifest(), self.runs)
        self.assertEqual(report["counts"]["invalid"], 1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["arms"]["control"]["attempts"], 17)  # not counted

    def test_unexpected_cells_block_but_quarantine_is_accounted(self):
        os.makedirs(os.path.join(self.runs, "control", "seed42", "stray-task"))
        report = controller.analyze(manifest(), self.runs)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["unexpected"]), 1)
        shutil.rmtree(os.path.join(self.runs, "control", "seed42", "stray-task"))
        os.makedirs(self.cell() + ".invalid-1")
        os.makedirs(self.cell() + ".aborted-1")
        report = controller.analyze(manifest(), self.runs)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["quarantined"]), 2)

    def test_clean_records_exit_zero(self):
        self.assertEqual(controller.main(["analyze", MANIFEST, "--runs", self.runs]), 0)


class TestZeroSuccess(Scratch):
    def fail_arm(self, arm: str) -> None:
        for _, seed, task in (
            k for k in controller.attempts(manifest()) if k[0] == arm
        ):
            edit(
                os.path.join(self.cell((arm, seed, task)), "results.json"),
                lambda d: d[0].update(run_successful=False, passed=False),
            )

    def test_control_without_success_is_explicit(self):
        self.fail_arm("control")
        report = controller.analyze(manifest(), self.runs)
        control = report["arms"]["control"]
        self.assertEqual(control["outcome"], "no_success")
        self.assertEqual(control["verified_tasks_per_hour"], 0.0)
        self.assertEqual(report["decision"]["status"], "ratio_undefined")
        self.assertIsNone(report["decision"]["met"])
        self.assertIn("control has 0 verified successes", report["decision"]["reason"])

    def test_treatment_without_success_does_not_win(self):
        self.fail_arm("verification")
        decision = controller.analyze(manifest(), self.runs)["decision"]
        self.assertEqual(decision["status"], "evaluated")
        self.assertIs(decision["met"], False)

    def test_neither_arm_succeeding_is_inconclusive(self):
        self.fail_arm("control")
        self.fail_arm("verification")
        report = controller.analyze(manifest(), self.runs)
        self.assertEqual(report["decision"]["status"], "inconclusive")
        self.assertTrue(report["complete"])


FAKE_LAUNCHER = textwrap.dedent(
    """
    import os, shutil, sys
    arm, records, mode = sys.argv[1:4]
    args = sys.argv[4:]
    stage = args[args.index("--run-dir") + 1]
    seed = args[args.index("--seed") + 1]
    task = args[args.index("--task") + 1]
    shutil.copytree(f"{records}/{arm}/seed{seed}/{task}", stage)
    if mode == "bad-prompt":
        import json
        p = os.path.join(stage, "exp2_policy.json")
        d = json.load(open(p)); d["prompt_sha256"] = "0" * 64
        json.dump(d, open(p, "w"))
    sys.exit(1)  # a failed task exits non-zero by design
    """
)


class TestRun(Scratch):
    def runnable(self, mode: str = "ok") -> dict[str, Any]:
        script = os.path.join(self.tmp, "launcher.py")
        with open(script, "w") as f:
            f.write(FAKE_LAUNCHER)
        m = copy.deepcopy(manifest())
        m["frozen"] = False
        for arm in m["arms"]:
            m["arms"][arm]["launch"] = [sys.executable, script, arm, RECORDS, mode]
        return m

    def test_frozen_manifest_is_never_run(self):
        self.assertEqual(controller.run(manifest(), self.runs, live=False), 2)

    def test_resume_skips_valid_cells_and_fills_the_rest(self):
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        before = {
            key: pathlib.Path(self.cell(key), "results.json").read_bytes()
            for key in controller.attempts(manifest())
            if key != last
        }
        self.assertEqual(controller.run(self.runnable(), self.runs, live=False), 0)
        self.assertTrue(os.path.exists(os.path.join(self.cell(last), "results.json")))
        self.assertTrue(os.path.exists(os.path.join(self.cell(last), "attempt.json")))
        self.assertFalse(os.path.exists(self.cell(last) + ".partial"))
        for key, blob in before.items():
            with open(os.path.join(self.cell(key), "results.json"), "rb") as f:
                self.assertEqual(f.read(), blob, f"{key} was rewritten")

    def test_crashed_attempt_is_kept_as_evidence(self):
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        os.makedirs(self.cell(last) + ".partial")
        with open(os.path.join(self.cell(last) + ".partial", "note"), "w") as f:
            f.write("half done")
        self.assertEqual(controller.run(self.runnable(), self.runs, live=False), 0)
        kept = os.path.join(self.cell(last) + ".aborted-1", "note")
        self.assertTrue(os.path.exists(kept))

    def test_invalid_attempt_halts_and_is_quarantined(self):
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        code = controller.run(self.runnable("bad-prompt"), self.runs, live=False)
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(self.cell(last)))
        self.assertTrue(os.path.isdir(self.cell(last) + ".invalid-1"))

    def test_existing_invalid_cell_stops_before_launching(self):
        edit(
            os.path.join(self.cell(), "exp2_policy.json"),
            lambda d: d.update(prompt_sha256="0" * 64),
        )
        shutil.rmtree(self.cell(("verification", 44, "refactor-validators")))
        self.assertEqual(controller.run(self.runnable(), self.runs, live=False), 2)
        self.assertFalse(
            os.path.exists(self.cell(("verification", 44, "refactor-validators")))
        )

    def test_promote_never_overwrites_a_cell(self):
        stage = self.cell() + ".partial"
        shutil.copytree(self.cell(), stage)
        problems = controller.promote(manifest(), CELL, stage, self.cell())
        self.assertTrue(any("overwrite" in p for p in problems))
        self.assertTrue(os.path.isdir(self.cell() + ".invalid-1"))
        self.assertEqual(self.problems_of(self.cell()), [])

    def problems_of(self, cell: str) -> list[str]:
        return controller.validate_attempt(manifest(), *CELL, cell)

    def test_arm_without_a_provenance_writing_entry_is_refused(self):
        m = self.runnable()
        self.assertEqual(controller.preflight(m), [])
        del m["arms"]["control"]["launch"]
        self.assertTrue(any("no launch entry" in p for p in controller.preflight(m)))
        self.assertEqual(controller.run(m, self.runs, live=False), 2)

    def test_preflight_detects_fixture_drift(self):
        self.assertEqual(controller.preflight(manifest()), [])
        m = copy.deepcopy(manifest())
        m["benchmark"]["suite"] = "v0.5"  # v0.5 replaced two of these tasks
        self.assertEqual(len(controller.preflight(m)), 2)
        m["benchmark"]["suite"] = "v0.4"
        m["tasks"] = [*m["tasks"], "no-such-task"]
        self.assertTrue(controller.preflight(m))


if __name__ == "__main__":
    unittest.main()
