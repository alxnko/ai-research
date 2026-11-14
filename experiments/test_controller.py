"""Offline tests of the experiment controller (no Ollama, GPU or network).

The reference data is `experiments/exp2_records/`: byte-for-byte copies of the
36 EXP-2 `config.json`/`exp2_policy.json`/`results.json` files (hashes in
SHA256SUMS). Every defect test runs on a throw-away copy, with a clean twin.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
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
from unittest import mock

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
    replaced = change(doc)
    if isinstance(replaced, dict | list):  # a change may swap the whole document
        doc = replaced
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


SYSTEM_PIN, SOURCE_PIN = "2" * 64, "1" * 64


def runnable_manifest() -> dict[str, Any]:
    """The EXP-2 manifest as a runnable one (adds the launcher-era identity pins)."""
    m = copy.deepcopy(manifest())
    m["frozen"] = False
    m["benchmark"] = {
        **m["benchmark"],
        "source_files": ["bench/bench.py"],
        "source_sha256": SOURCE_PIN,
    }
    for arm in m["arms"].values():
        arm["system_prompt_sha256"] = SYSTEM_PIN
    return m


def launcher_fields(m: dict[str, Any], key: tuple[str, int, str]) -> dict[str, Any]:
    """What the generic launcher adds to a provenance record."""
    return {
        "system_prompt_sha256": SYSTEM_PIN,
        "model_calls": 1,
        "source_sha256": SOURCE_PIN,
        "source_verification": "git+filesystem",
        "attempt_id": controller.attempt_id(m, *key),
    }


class Scratch(unittest.TestCase):
    """A private copy of the EXP-2 records for each test."""

    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.mkdtemp(
            prefix="controller-test-",
            dir="/var/tmp" if os.path.isdir("/var/tmp") else None,
        )
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = os.path.join(self.tmp, "runs")
        shutil.copytree(RECORDS, self.runs)
        # Hash-bind against this test's own (editable) copy, not the committed one.
        sums = os.path.join(self.runs, "SHA256SUMS")
        real = globals()["manifest"]
        self.enterContext(
            mock.patch.dict(
                globals(),
                {"manifest": lambda: {**real(), "integrity": sums}},
            )
        )

    def cell(self, key: Key = CELL) -> str:
        return controller.cell_dir(self.runs, *key)

    def rewrite(self, key: Key, name: str, change) -> None:
        """Edit a record file and re-bless it in SHA256SUMS (a deliberate,
        internally consistent edit; the binding tests tamper without this)."""
        path = os.path.join(self.cell(key), name)
        edit(path, change)
        sums = os.path.join(self.runs, "SHA256SUMS")
        with open(sums, encoding="utf-8") as f:
            lines = f.read().splitlines()
        rel = f"{key[0]}/seed{key[1]}/{key[2]}/{name}"
        digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        lines = [f"{digest}  {rel}" if ln.endswith("  " + rel) else ln for ln in lines]
        pathlib.Path(sums).write_text("\n".join(lines) + "\n")


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
        # the controller accepts the real directory itself (aborted/, logs/ and all)
        report = controller.analyze(manifest(), originals)
        self.assertTrue(report["complete"], report["unexpected"])
        self.assertEqual(report["decision"]["ratio"], 1.51)

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
        moved = copy.deepcopy(m)
        moved["arms"]["control"]["launch"] = ["/elsewhere/python"]
        moved["aborted"] = []
        moved["runs_dir"] = "runs/other"
        self.assertEqual(
            controller.attempt_id(m, *CELL), controller.attempt_id(moved, *CELL)
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
            "frozen not stated": drop("frozen"),
            "frozen without integrity": drop("integrity"),
            "absolute integrity path": setpath(["integrity"], "/etc/SHA256SUMS"),
            "min_ratio as text": setpath(["analysis", "min_ratio"], "1.25"),
            "min_ratio zero": setpath(["analysis", "min_ratio"], 0),
            "same control and treatment": setpath(["analysis", "treatment"], "control"),
            "path traversal in task": setpath(["tasks"], ["../../x"]),
            "absolute runs_dir": setpath(["runs_dir"], "/etc"),
            "launch not a list": setpath(["arms", "control", "launch"], "python"),
            "float seed": setpath(["seeds"], [42.5]),
        }
        for label, change in defects.items():
            m = copy.deepcopy(manifest())
            change(m)
            self.assertTrue(controller.validate_manifest(m), label)


class TestManifestRobustness(unittest.TestCase):
    JUNK = ({}, [], [{}], [[]], None, 1, "x", True, float("nan"), {"a": 1})

    def test_malformed_values_are_reported_never_raised(self):
        base = manifest()
        paths = [(field,) for field in base] + [
            ("settings", "ctx"),
            ("settings", "sampling"),
            ("settings", "sampling", "top_p"),
            ("arms", "control"),
            ("arms", "control", "launch"),
            ("analysis", "min_ratio"),
            ("model", "tag"),
            ("benchmark", "suite"),
        ]
        for path in paths:
            for junk in self.JUNK:
                m = copy.deepcopy(base)
                node = m
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = junk
                try:
                    problems = controller.validate_manifest(m)
                except Exception as exc:  # the contract: never raise
                    self.fail(f"{path}={junk!r} raised {type(exc).__name__}: {exc}")
                expected = base_value(base, path)
                numeric = all(
                    isinstance(v, int | float) and not isinstance(v, bool)
                    for v in (junk, expected)
                )
                if (
                    expected is not None
                    and not numeric
                    and type(junk) is not type(expected)
                ):
                    self.assertTrue(problems, f"{path}={junk!r} was accepted")

    def test_ranges_and_non_finite_numbers(self):
        def with_(path, value):
            m = copy.deepcopy(manifest())
            node = m
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = value
            return controller.validate_manifest(m)

        bad = {
            ("settings", "sampling", "top_p"): [0.0, 1.5, -0.1, float("nan")],
            ("settings", "sampling", "temperature"): [-1.0, float("inf")],
            ("settings", "sampling", "repeat_penalty"): [0.0, float("inf")],
            ("settings", "sampling", "top_k"): [-1],
            ("settings", "steps"): [0, -3],
            ("settings", "max_wall"): [0],
            ("settings", "ctx"): [0],
            ("settings", "ngpu"): [-1],
            ("analysis", "min_ratio"): [float("inf"), float("nan"), -1.0],
        }
        for path, values in bad.items():
            for value in values:
                self.assertTrue(with_(path, value), f"{path}={value!r} accepted")
        good = {
            ("settings", "sampling", "top_p"): [0.95, 1.0],
            ("settings", "steps"): [1, 20],
        }
        for path, values in good.items():
            for value in values:
                self.assertEqual(with_(path, value), [], f"{path}={value!r}")


def base_value(base: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = base
    for part in path:
        node = node.get(part) if isinstance(node, dict) else None
    return node


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
            "other experiment": (policy, lambda d: d.update(experiment="EXP-9")),
            "other core commit": (
                policy,
                lambda d: d.update(benchmark_core_commit="0"),
            ),
            "fingerprint not an object": (
                policy,
                lambda d: d.update(model_fingerprint="sha256:d8b0"),
            ),
            "config is a list": (config, lambda _d: [1]),
            "results rows are lists": (results, lambda d: d.__setitem__(0, [1])),
            "missing required key": (results, lambda d: d[0].pop("tokens")),
            "other actuator": (results, lambda d: d[0].update(actuator="text")),
            "tokens as text": (results, lambda d: d[0].update(tokens="93812")),
            "peak vram as text": (results, lambda d: d[0].update(peak_vram_mib="big")),
            "invalid actions negative": (
                results,
                lambda d: d[0].update(invalid_actions=-1),
            ),
            "infinite wall time": (
                results,
                lambda d: d[0].update(total_wall_s=float("inf")),
            ),
            "nan wall time": (
                results,
                lambda d: d[0].update(total_wall_s=float("nan")),
            ),
            "passed not boolean": (results, lambda d: d[0].update(passed="yes")),
            "1 for True": (results, lambda d: d[0].update(strict_isolation=1)),
            "float for int": (config, lambda d: d.update(ctx=16384.0)),
            "other task in config": (
                config,
                lambda d: d.update(task="refactor-parser"),
            ),
            "other seed": (config, lambda d: d["sampling"].update(seed=43)),
            "other model tag": (config, lambda d: d.update(model="qwen3.5:4b")),
            "other budget": (config, lambda d: d.update(max_wall=1200)),
        }
        for label, (name, change) in defects.items():
            # A fresh copy per defect, edited and re-blessed in the hash list, so
            # only the content checks (not the hash binding) can reject it.
            shutil.rmtree(self.runs)
            shutil.copytree(RECORDS, self.runs)
            self.assertEqual(self.problems(), [], label)
            self.rewrite(CELL, name, change)
            found = self.problems()
            self.assertTrue(found, f"{label} was accepted")
            self.assertFalse(
                any("integrity file" in p for p in found), f"{label}: {found}"
            )

    def test_task_absent_from_the_suite_is_rejected(self):
        edit(
            os.path.join(self.cell(), "results.json"),
            lambda d: d[0].pop("task_digest"),
        )
        with mock.patch.object(controller, "_suite_pins", lambda _m: {}):
            found = controller.validate_attempt(manifest(), *CELL, self.cell())
        self.assertTrue(any("not in the suite" in p for p in found), found)

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
            self.rewrite(
                (arm, seed, task),
                "results.json",
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


class TestBinding(Scratch):
    """A record cannot be moved between cells or edited after the fact."""

    def test_results_copied_from_another_arm_are_rejected(self):
        other = ("verification", 42, "backend-auth-token")
        shutil.copyfile(
            os.path.join(self.cell(other), "results.json"),
            os.path.join(self.cell(), "results.json"),
        )
        found = controller.validate_attempt(manifest(), *CELL, self.cell())
        self.assertTrue(any("integrity file" in p for p in found), found)
        self.assertFalse(controller.analyze(manifest(), self.runs)["complete"])

    def upgraded(self, m: dict[str, Any], key: Key) -> None:
        edit(
            os.path.join(self.cell(key), "exp2_policy.json"),
            lambda d: d.update(launcher_fields(m, key)),
        )

    def test_edit_after_promotion_is_rejected_for_new_runs(self):
        m = runnable_manifest()
        os.remove(os.path.join(self.runs, "SHA256SUMS"))
        self.upgraded(m, CELL)
        controller._write_attempt(m, CELL, self.cell(), durable=False)
        self.assertEqual(controller.validate_attempt(m, *CELL, self.cell()), [])
        edit(
            os.path.join(self.cell(), "results.json"),
            lambda d: d[0].update(total_wall_s=1.0),
        )
        found = controller.validate_attempt(m, *CELL, self.cell())
        self.assertTrue(any("recorded hash" in p for p in found), found)

    def test_frozen_records_without_their_integrity_file_fail_closed(self):
        m = {**manifest(), "integrity": os.path.join(self.tmp, "missing-sums")}
        found = controller.validate_attempt(m, *CELL, self.cell())
        self.assertTrue(any("integrity file" in p for p in found), found)
        os.makedirs(os.path.join(self.tmp, "sums-dir"))
        m["integrity"] = os.path.join(self.tmp, "sums-dir")
        self.assertTrue(controller.validate_attempt(m, *CELL, self.cell()))

    def test_new_run_without_attempt_json_is_rejected(self):
        m = runnable_manifest()
        found = controller.validate_attempt(m, *CELL, self.cell())
        self.assertTrue(any("attempt.json" in p for p in found), found)

    def test_attempt_json_of_another_cell_is_rejected(self):
        m = runnable_manifest()
        os.remove(os.path.join(self.runs, "SHA256SUMS"))
        other = ("control", 43, "backend-auth-token")
        self.upgraded(m, CELL)
        self.upgraded(m, other)
        controller._write_attempt(m, other, self.cell(other), durable=False)
        shutil.copyfile(
            os.path.join(self.cell(other), "attempt.json"),
            os.path.join(self.cell(), "attempt.json"),
        )
        found = controller.validate_attempt(m, *CELL, self.cell())
        self.assertTrue(any("attempt_id" in p for p in found), found)

    def test_symlinked_records_are_rejected(self):
        real = os.path.join(self.tmp, "elsewhere.json")
        shutil.copyfile(os.path.join(self.cell(), "results.json"), real)
        os.remove(os.path.join(self.cell(), "results.json"))
        os.symlink(real, os.path.join(self.cell(), "results.json"))
        found = controller.validate_attempt(manifest(), *CELL, self.cell())
        self.assertTrue(any("symlink" in p for p in found), found)


class TestLauncherProvenance(Scratch):
    """Records of runnable manifests must carry the launcher's observations."""

    def setUp(self) -> None:
        super().setUp()
        self.m = runnable_manifest()
        os.remove(os.path.join(self.runs, "SHA256SUMS"))
        edit(
            os.path.join(self.cell(), "exp2_policy.json"),
            lambda d: d.update(launcher_fields(self.m, CELL)),
        )
        self.bless()

    def bless(self) -> None:
        controller._write_attempt(self.m, CELL, self.cell(), durable=False)

    def problems(self) -> list[str]:
        return controller.validate_attempt(self.m, *CELL, self.cell())

    def test_clean_launcher_record_is_valid(self):
        self.assertEqual(self.problems(), [])

    def test_each_observation_defect_is_detected(self):
        policy = os.path.join(self.cell(), "exp2_policy.json")
        defects = {
            "other system prompt observed": {"system_prompt_sha256": "3" * 64},
            "no model call": {"model_calls": 0},
            "model calls as text": {"model_calls": "1"},
            "other source identity": {"source_sha256": "4" * 64},
            "source not verified against the tag": {
                "source_verification": "digest-only"
            },
            "another attempt's provenance": {"attempt_id": "0" * 16},
        }
        good = json.loads(pathlib.Path(policy).read_text())
        for label, change in defects.items():
            pathlib.Path(policy).write_text(json.dumps({**good, **change}))
            self.bless()  # re-bless: only the content checks may reject it
            found = self.problems()
            self.assertTrue(found, f"{label} was accepted")
            self.assertFalse(any("recorded hash" in p for p in found), label)
        for key in ("system_prompt_sha256", "model_calls", "source_sha256"):
            pathlib.Path(policy).write_text(
                json.dumps({k: v for k, v in good.items() if k != key})
            )
            self.bless()
            self.assertTrue(self.problems(), f"missing {key} was accepted")
        pathlib.Path(policy).write_text(json.dumps(good))
        self.bless()
        self.assertEqual(self.problems(), [])


class TestDecisionRule(unittest.TestCase):
    """Each condition of the registered rule is decisive on its own, and the
    decision uses exact (never display-rounded) throughput."""

    @staticmethod
    def stats(successes, wall, errors=0, timeouts=0):
        return {
            "successes": successes,
            "wall_exact": wall,
            "inference_errors": errors,
            "timeouts": timeouts,
        }

    def decide(self, control, treatment):
        return controller.decide(
            manifest(), {"control": control, "verification": treatment}
        )

    def test_boundaries(self):
        # equal successes, so ratio = control_wall / treatment_wall
        base = self.stats(6, 1000.0)
        cases = {
            "exactly 1.25x": (self.stats(6, 800.0), True),
            "just below 1.25x": (self.stats(6, 800.001), False),
            "just above 1.25x": (self.stats(6, 799.999), True),
            "fewer successes": (self.stats(5, 400.0), False),
            "equal successes": (self.stats(6, 400.0), True),
            "more inference errors": (self.stats(9, 400.0, errors=1), False),
            "more timeouts": (self.stats(9, 400.0, timeouts=1), False),
        }
        for label, (treatment, met) in cases.items():
            result = self.decide(base, treatment)
            self.assertEqual(result["status"], "evaluated", label)
            self.assertIs(result["met"], met, label)
        # errors/timeouts equal to control's are not a regression
        both = self.decide(self.stats(6, 1000.0, 1, 1), self.stats(9, 400.0, 1, 1))
        self.assertIs(both["met"], True)

    def test_display_rounding_cannot_decide(self):
        # 3600*s/wall: control 4.004/h, treatment 5.004/h. Both display as
        # 4.00 and 5.00 (ratio 1.25 when rounded) but the true ratio is 1.24975.
        control = controller.summarize(
            [{"run_successful": True, "total_wall_s": 3600 / 4.004}]
        )
        treatment = controller.summarize(
            [{"run_successful": True, "total_wall_s": 3600 / 5.004}]
        )
        self.assertEqual(control["verified_tasks_per_hour"], 4.0)
        self.assertEqual(treatment["verified_tasks_per_hour"], 5.0)
        result = self.decide(control, treatment)
        self.assertIs(result["met"], False)
        self.assertLess(result["ratio_exact"], 1.25)
        # the clean twin: a real 1.25x passes
        passing = controller.summarize(
            [{"run_successful": True, "total_wall_s": 3600 / 5.005}]
        )
        self.assertIs(self.decide(control, passing)["met"], True)

    def test_zero_recorded_wall_time_is_reported_not_raised(self):
        for control, treatment in [
            (self.stats(3, 0.0), self.stats(3, 100.0)),
            (self.stats(3, 100.0), self.stats(3, 0.0)),
            (self.stats(3, 100.0), self.stats(0, 0.0)),
        ]:
            result = self.decide(control, treatment)
            self.assertEqual(result["status"], "ratio_undefined")
            self.assertIsNone(result["met"])

    def test_summarize_zero_wall(self):
        row = {"run_successful": True, "total_wall_s": 0, "tokens": 1}
        self.assertIsNone(controller.summarize([row])["verified_tasks_per_hour"])
        failed = {"run_successful": False, "total_wall_s": 10, "tokens": 1}
        self.assertEqual(controller.summarize([failed])["verified_tasks_per_hour"], 0.0)


class TestScanStrictness(Scratch):
    def test_operator_directories_beside_the_arms_are_ignored(self):
        # the real runs/exp2 holds aborted attempts, logs and a summary log
        os.makedirs(os.path.join(self.runs, "aborted", "killed-seed42", "r0"))
        os.makedirs(os.path.join(self.runs, "logs"))
        pathlib.Path(self.runs, "logs", "a.log").write_text("x")
        pathlib.Path(self.runs, "attempts.log").write_text("x")
        report = controller.analyze(manifest(), self.runs)
        self.assertTrue(report["complete"], report["unexpected"])
        self.assertEqual(report["decision"]["status"], "evaluated")

    def test_stray_quarantine_lookalike_is_unexpected(self):
        stray = os.path.join(self.runs, "control", "seed42", "new-task.partial")
        os.makedirs(stray)
        report = controller.analyze(manifest(), self.runs)
        self.assertEqual(report["unexpected"], [stray])
        self.assertFalse(report["complete"])

    def test_stray_entries_inside_an_arm_are_unexpected(self):
        os.makedirs(os.path.join(self.runs, "control", "scratch"))
        report = controller.analyze(manifest(), self.runs)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["unexpected"]), 1)

    def test_quarantined_attempts_are_listed_in_the_report(self):
        os.makedirs(self.cell() + ".invalid-1")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            controller.main(["analyze", MANIFEST, "--runs", self.runs])
        self.assertIn("QUARANTINED", out.getvalue())


FAKE_LAUNCHER = textwrap.dedent(
    """
    import os, shutil, sys
    arm, records, mode, meta = sys.argv[1:5]
    args = sys.argv[5:]
    stage = args[args.index("--run-dir") + 1]
    seed = args[args.index("--seed") + 1]
    task = args[args.index("--task") + 1]
    shutil.copytree(f"{records}/{arm}/seed{seed}/{task}", stage, dirs_exist_ok=True)
    import json
    p = os.path.join(stage, "exp2_policy.json")
    d = json.load(open(p))
    d.update(json.load(open(meta))[f"{arm}/{seed}/{task}"])
    if mode == "bad-prompt":
        d["prompt_sha256"] = "0" * 64
    json.dump(d, open(p, "w"))
    sys.exit(1)  # a failed task exits non-zero by design
    """
)


class TestRun(Scratch):
    def runnable(self, mode: str = "ok") -> dict[str, Any]:
        script = os.path.join(self.tmp, "launcher.py")
        with open(script, "w") as f:
            f.write(FAKE_LAUNCHER)
        m = runnable_manifest()
        meta = os.path.join(self.tmp, "meta.json")
        pathlib.Path(meta).write_text(
            json.dumps(
                {
                    "/".join(map(str, key)): launcher_fields(m, key)
                    for key in controller.attempts(m)
                }
            )
        )
        for arm in m["arms"]:
            m["arms"][arm]["launch"] = [
                sys.executable,
                script,
                arm,
                RECORDS,
                mode,
                meta,
            ]
        # new runs bind by attempt.json, not by a SHA256SUMS
        pathlib.Path(self.runs, "SHA256SUMS").unlink(missing_ok=True)
        for key in controller.attempts(m):
            if os.path.isdir(self.cell(key)):
                edit(
                    os.path.join(self.cell(key), "exp2_policy.json"),
                    lambda d, key=key: d.update(launcher_fields(m, key)),
                )
                controller._write_attempt(m, key, self.cell(key), durable=False)
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

    def test_crashed_attempt_blocks_the_next_run_and_is_kept(self):
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        stage = self.cell(last) + ".partial"
        os.makedirs(stage)
        pathlib.Path(stage, "note").write_text("half done")
        m = self.runnable()
        self.assertEqual(controller.run(m, self.runs, live=False), 2)  # no re-roll
        self.assertEqual(pathlib.Path(stage, "note").read_text(), "half done")
        self.assertFalse(os.path.exists(self.cell(last)))
        # a person resolves it by filing it as evidence beside the matrix
        kept = os.path.join(self.runs, "aborted", "verification-seed44-partial")
        os.makedirs(os.path.dirname(kept))
        os.rename(stage, kept)
        self.assertEqual(controller.run(m, self.runs, live=False), 0)
        self.assertEqual(pathlib.Path(kept, "note").read_text(), "half done")
        self.assertTrue(controller.analyze(m, self.runs)["complete"])

    def test_invalid_attempt_halts_then_blocks_until_resolved(self):
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        code = controller.run(self.runnable("bad-prompt"), self.runs, live=False)
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(self.cell(last)))
        self.assertTrue(os.path.isdir(self.cell(last) + ".invalid-1"))
        # re-running must not silently re-roll the failed attempt
        self.assertEqual(controller.run(self.runnable(), self.runs, live=False), 2)
        self.assertFalse(os.path.exists(self.cell(last)))

    def silent_launcher(self, counter: str) -> dict[str, Any]:
        """An arm whose launcher counts its calls and writes nothing."""
        script = os.path.join(self.tmp, "silent.py")
        with open(script, "w") as f:
            f.write(f"import sys\nopen({counter!r}, 'a').write('x')\nsys.exit(1)\n")
        m = self.runnable()
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        m["arms"]["verification"]["launch"] = [sys.executable, script]
        return m

    def test_launcher_that_writes_nothing_cannot_be_silently_retried(self):
        counter = os.path.join(self.tmp, "calls")
        m = self.silent_launcher(counter)
        last = ("verification", 44, "refactor-validators")
        self.assertEqual(controller.run(m, self.runs, live=False), 3)
        self.assertEqual(pathlib.Path(counter).read_text(), "x")
        self.assertTrue(os.path.isdir(self.cell(last) + ".partial"))
        # the second invocation must refuse, not launch again
        self.assertEqual(controller.run(m, self.runs, live=False), 2)
        self.assertEqual(pathlib.Path(counter).read_text(), "x")

    def test_launcher_that_cannot_start_leaves_evidence(self):
        m = self.runnable()
        last = ("verification", 44, "refactor-validators")
        shutil.rmtree(self.cell(last))
        m["arms"]["verification"]["launch"] = [os.path.join(self.tmp, "no-such-bin")]
        self.assertEqual(controller.run(m, self.runs, live=False), 3)
        note = pathlib.Path(self.cell(last) + ".partial", "controller-note.txt")
        self.assertIn("could not start", note.read_text())
        self.assertEqual(controller.run(m, self.runs, live=False), 2)

    def test_directory_without_a_result_is_refused_before_launching(self):
        last = ("verification", 44, "refactor-validators")
        m = self.runnable()
        os.remove(os.path.join(self.cell(last), "results.json"))
        self.assertEqual(controller.run(m, self.runs, live=False), 2)
        self.assertFalse(os.path.exists(self.cell(last) + ".invalid-1"))

    def test_second_controller_is_locked_out(self):
        m = self.runnable()
        with open(os.path.join(self.runs, ".controller.lock"), "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(controller.run(m, self.runs, live=False), 2)

    def test_stray_entries_stop_a_run(self):
        os.makedirs(os.path.join(self.runs, "control", "seed42", "extra-task"))
        self.assertEqual(controller.run(self.runnable(), self.runs, live=False), 2)

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

    def test_arm_without_a_launcher_needs_the_manifest_path(self):
        m = self.runnable()
        self.assertEqual(controller.preflight(m), [])
        del m["arms"]["control"]["launch"]
        # the generic launcher is started with the manifest's path; without it
        # the arm has no way to write provenance, so nothing is launched
        problems = controller._preflight_run(m, self.runs, False, "")
        self.assertTrue(any("no launch entry" in p for p in problems), problems)
        self.assertEqual(controller.run(m, self.runs, live=False), 2)
        self.assertEqual(
            controller._preflight_run(m, self.runs, False, "/some/manifest.json"), []
        )

    def test_live_model_identity_is_checked_against_all_three_digests(self):
        good = {k: v for k, v in manifest()["model"].items() if k.endswith("_digest")}
        with mock.patch.object(controller.engine, "model_fingerprint", lambda _t: good):
            self.assertEqual(controller.preflight(manifest(), live=True), [])
        for key in good:
            bad = {**good, key: "sha256:other"}
            with mock.patch.object(
                controller.engine, "model_fingerprint", lambda _t, bad=bad: bad
            ):
                self.assertTrue(controller.preflight(manifest(), live=True), key)

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
