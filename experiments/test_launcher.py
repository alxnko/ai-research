"""Offline tests: code identity (source_identity) and the generic launcher.

Source-identity tests run on throw-away git repositories, so drift cases never
touch this checkout. The launcher tests run the real controller -> launcher ->
bench (mock model) -> validation -> promotion path against the real `v0.5`
release (skipped, and reported as such, if that tag is not fetched).
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
import subprocess
import sys
import tempfile
import types
import unittest
from typing import Any
from unittest import mock

from bench import bench as engine
from experiments import controller, launcher, source_identity

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "v0.5"
SOURCE_FILES = [
    "bench/__init__.py",
    "bench/bench.py",
    "bench/classifier.py",
    "bench/telemetry.py",
]
OLD_SENTENCE = "Ensure all changes are verified before finishing."
NEW_SENTENCE = "Before finishing, verify the requested behavior with targeted checks."
TASK = "debugging-off-by-one"


def tmpdir() -> str:
    return tempfile.mkdtemp(
        prefix="launcher-test-", dir="/var/tmp" if os.path.isdir("/var/tmp") else None
    )


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(root: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


class Quiet(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tmpdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class TempRepo(Quiet):
    """A tiny benchmark release: engine files, one suite, one task, tag vT."""

    def setUp(self) -> None:
        super().setUp()
        self.root = os.path.join(self.tmp, "repo")
        files = {
            "bench/__init__.py": "",
            "bench/bench.py": "ENGINE = 1\n",
            "bench/classifier.py": "CLASSIFIER = 1\n",
            "bench/telemetry.py": "TELEMETRY = 1\n",
            "bench/tasks/t1/repo/a.py": "def f():\n    return 1\n",
            "bench/tasks/t1/tests/test_a.py": "ASSERT = 1\n",
            "experiments/controller.py": "CONTROLLER = 1\n",
        }
        for rel, text in files.items():
            self.put(rel, text)
        digest = source_identity.dir_digest(os.path.join(self.root, "bench/tasks/t1"))
        self.put(
            "bench/suites/s.json", json.dumps({"suite": "s", "tasks": {"t1": digest}})
        )
        git(self.root, "init", "-q")
        git(self.root, "config", "user.email", "t@example.invalid")
        git(self.root, "config", "user.name", "t")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "release")
        git(self.root, "tag", "-a", "vT", "-m", "release")
        reg = source_identity.registration("vT", "s", SOURCE_FILES, self.root)
        self.manifest: dict[str, Any] = {
            "benchmark": {"tag": "vT", "suite": "s", **reg},
            "tasks": ["t1"],
        }

    def put(self, rel: str, text: str) -> None:
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pathlib.Path(path).write_text(text)

    def problems(self, manifest: dict[str, Any] | None = None, **kw) -> list[str]:
        return source_identity.verify_source(manifest or self.manifest, self.root, **kw)

    def reregister(self) -> dict[str, Any]:
        """A manifest whose pins were recomputed from the (tampered) disk."""
        m = copy.deepcopy(self.manifest)
        m["benchmark"]["source_sha256"] = source_identity.files_digest(
            self.root, source_identity.source_paths(m)
        )
        return m


class TestSourceIdentity(TempRepo):
    def test_clean_release_is_verified(self):
        self.assertEqual(self.problems(), [])
        self.assertEqual(self.problems(use_git=False), [])

    def test_head_may_move_on_while_the_benchmark_stays_identical(self):
        self.put("experiments/controller.py", "CONTROLLER = 2\n")
        self.put("docs/new.md", "new\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "later controller work")
        self.assertNotEqual(
            git(self.root, "rev-parse", "HEAD"), self.manifest["benchmark"]["commit"]
        )
        self.assertEqual(self.problems(), [])

    def test_each_source_file_edit_is_detected(self):
        for rel in [*SOURCE_FILES[1:], "bench/suites/s.json"]:
            original = pathlib.Path(self.root, rel).read_text()
            if rel.endswith(".json"):  # keep it valid JSON, so only content drifts
                drifted = json.dumps({**json.loads(original), "note": "drift"})
            else:
                drifted = original + "# drift\n"
            pathlib.Path(self.root, rel).write_text(drifted)
            found = self.problems()
            self.assertTrue(any("registered digest" in p for p in found), (rel, found))
            self.assertTrue(
                any("differs from the tag" in p for p in found), (rel, found)
            )
            pathlib.Path(self.root, rel).write_text(original)
        self.assertEqual(self.problems(), [])

    def test_committed_drift_after_the_tag_is_still_drift(self):
        self.put("bench/bench.py", "ENGINE = 2\n")
        git(self.root, "commit", "-q", "-am", "engine change after the release")
        self.assertTrue(self.problems())

    def test_repinned_tampering_is_caught_by_the_tag(self):
        # the digest pin can be recomputed from tampered files; the tag cannot
        self.put("bench/bench.py", "ENGINE = 'tampered'\n")
        tampered = self.reregister()
        self.assertEqual(self.problems(tampered, use_git=False), [])  # digest alone
        found = self.problems(tampered)
        self.assertTrue(
            any("bench/bench.py differs from the tag" in p for p in found), found
        )

    def test_fixture_drift_is_detected(self):
        self.put("bench/tasks/t1/tests/test_a.py", "ASSERT = 2\n")
        found = self.problems()
        self.assertTrue(
            any("fixtures differ from the suite pin" in p for p in found), found
        )
        self.assertTrue(self.problems(use_git=False))

    def test_stray_file_in_a_fixture_is_detected(self):
        self.put("bench/tasks/t1/__pycache__/a.pyc", "x")
        self.assertTrue(any("fixtures differ" in p for p in self.problems()))

    def test_repinned_fixture_drift_is_caught_by_the_tag(self):
        self.put("bench/tasks/t1/tests/test_a.py", "ASSERT = 2\n")
        new = source_identity.dir_digest(os.path.join(self.root, "bench/tasks/t1"))
        self.put(
            "bench/suites/s.json", json.dumps({"suite": "s", "tasks": {"t1": new}})
        )
        found = self.problems(self.reregister())
        self.assertTrue(
            any(
                "at the tag differs" in p or "differs from the tag" in p for p in found
            ),
            found,
        )

    def test_wrong_registered_digest_is_detected(self):
        m = copy.deepcopy(self.manifest)
        m["benchmark"]["source_sha256"] = "0" * 64
        self.assertTrue(any("registered digest" in p for p in self.problems(m)))

    def test_tag_problems(self):
        m = copy.deepcopy(self.manifest)
        m["benchmark"]["tag"] = "no-such-tag"
        self.assertTrue(any("not found" in p for p in self.problems(m)))
        self.assertEqual(self.problems(m, use_git=False), [])  # offline: not evaluated
        # the tag is moved to another commit: same files, different commit
        self.put("docs/x.md", "x\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "other")
        git(self.root, "tag", "-f", "-a", "vT", "-m", "moved")
        self.assertTrue(any("resolves to" in p for p in self.problems()))

    def test_symlinked_or_missing_source_is_detected(self):
        target = os.path.join(self.tmp, "elsewhere.py")
        pathlib.Path(target).write_text("CLASSIFIER = 1\n")
        cls = os.path.join(self.root, "bench/classifier.py")
        os.remove(cls)
        os.symlink(target, cls)
        self.assertTrue(any("symlink" in p for p in self.problems()))
        os.remove(cls)
        self.assertTrue(any("missing" in p for p in self.problems()))

    def test_manifest_without_identity_is_refused(self):
        m = copy.deepcopy(self.manifest)
        del m["benchmark"]["source_sha256"]
        self.assertTrue(any("no source identity" in p for p in self.problems(m)))

    def test_task_missing_from_the_suite_is_detected(self):
        m = copy.deepcopy(self.manifest)
        m["tasks"] = ["t1", "t2"]
        self.assertTrue(any("'t2' is not in the suite" in p for p in self.problems(m)))

    def test_release_whose_own_pin_disagrees_with_its_task_is_detected(self):
        # disk matches the pin, but the tagged task bytes do not: only the
        # comparison against the tag's task objects can see this
        self.put("bench/tasks/t1/tests/test_a.py", "ASSERT = 2\n")
        new = source_identity.dir_digest(os.path.join(self.root, "bench/tasks/t1"))
        self.put(
            "bench/suites/s.json", json.dumps({"suite": "s", "tasks": {"t1": new}})
        )
        git(self.root, "add", "bench/suites/s.json")  # the task file stays unstaged
        git(self.root, "commit", "-q", "-m", "inconsistent release")
        git(self.root, "tag", "-f", "-a", "vT", "-m", "inconsistent")
        reg = source_identity.registration("vT", "s", SOURCE_FILES, self.root)
        m = {"benchmark": {"tag": "vT", "suite": "s", **reg}, "tasks": ["t1"]}
        found = self.problems(m)
        self.assertEqual(found, ["task 't1' at the tag differs from the suite pin"])
        self.assertEqual(self.problems(m, use_git=False), [])

    def test_dir_digest_matches_the_engine_algorithm(self):
        task = os.path.join(controller.ROOT, "bench", "tasks", TASK)
        self.assertEqual(source_identity.dir_digest(task), engine.task_digest(TASK))


def have_release() -> bool:
    return source_identity.resolve_tag(controller.ROOT, TAG) is not None


class Release(Quiet):
    """A runnable manifest against the real v0.5 release, a fake Ollama store."""

    MODEL = "qwen3.5:4b-q4_K_M"

    def setUp(self) -> None:
        super().setUp()
        if not have_release():
            self.skipTest(f"tag {TAG} not available (SKIPPED, not verified)")
        self.home = os.path.join(self.tmp, "home")
        self.digests = self.fake_ollama(self.home, self.MODEL)
        env = {"HOME": self.home, "BENCH_MOCK_MODEL": "1"}
        self.enterContext(mock.patch.dict(os.environ, env))
        for name in (
            "SYSTEM_PROMPT_TOOLS",
            "SYSTEM_PROMPT_TEXT",
            "chat_stream",
            "_mock_chat_stream",
        ):
            self.enterContext(mock.patch.object(engine, name, getattr(engine, name)))
        self.runs = os.path.join(self.tmp, "runs")
        self.path = os.path.join(self.tmp, "manifest.json")
        real_run: Any = (
            subprocess.run
        )  # keep the engine's console output out of the log

        def quiet_run(cmd: Any, **kwargs: Any) -> Any:
            return real_run(cmd, **{**kwargs, "capture_output": True})

        self.enterContext(mock.patch.object(controller.subprocess, "run", quiet_run))

    @staticmethod
    def fake_ollama(home: str, model: str, weights: str = "w") -> dict[str, str]:
        """A real-layout Ollama store, so the engine's own fingerprint code runs."""
        name, _, tag = model.partition(":")
        store = os.path.join(home, ".ollama", "models")
        listing = os.path.join(
            store, "manifests-v2", "registry.ollama.ai", "library", name
        )
        os.makedirs(listing, exist_ok=True)
        os.makedirs(os.path.join(store, "blobs"), exist_ok=True)
        digests = {
            "manifest_digest": "sha256:" + sha("manifest"),
            "config_digest": "sha256:" + sha("config"),
            "weights_digest": "sha256:" + sha(weights),
        }
        blob = {
            "config": {"digest": digests["config_digest"]},
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": digests["weights_digest"],
                    "size": 123,
                }
            ],
        }
        pathlib.Path(listing, tag).write_text(
            json.dumps({"manifests": [{"digest": digests["manifest_digest"]}]})
        )
        blob_name = digests["manifest_digest"].replace(":", "-")
        pathlib.Path(store, "blobs", blob_name).write_text(json.dumps(blob))
        return digests

    def manifest(self) -> dict[str, Any]:
        base = controller.load_manifest(os.path.join(HERE, "manifests", "exp2.json"))
        m = copy.deepcopy(base)
        m.pop("integrity")
        m.pop("aborted")
        m.pop("provenance_file")
        m["frozen"] = False
        m["experiment"] = "TEST-LAUNCHER"
        m["tasks"], m["seeds"] = [TASK], [42]
        m["settings"]["strict_isolation"] = False
        m["model"] = {"tag": self.MODEL, **self.digests}
        reg = source_identity.registration(TAG, "v0.5", SOURCE_FILES)
        m["benchmark"] = {"tag": TAG, "suite": "v0.5", **reg}
        arms: dict[str, dict[str, Any]] = {
            "control": {},
            "verification": {"prompt_edit": {"replace": [OLD_SENTENCE, NEW_SENTENCE]}},
        }
        m["arms"] = arms
        for arm in m["arms"]:
            m["arms"][arm].update(launcher.prompt_hashes(m, arm))
        return m

    def write(self, m: dict[str, Any]) -> dict[str, Any]:
        pathlib.Path(self.path).write_text(json.dumps(m))
        return controller.load_manifest(self.path)

    def launch(
        self, m: dict[str, Any], arm: str, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        """The real launcher in a subprocess, as the controller starts it."""
        self.write(m)
        stage = os.path.join(self.tmp, f"stage-{arm}-{len(os.listdir(self.tmp))}")
        os.makedirs(stage)
        self.stage = stage
        argv = controller.bench_argv(m, 42, TASK)
        proc = subprocess.run(
            [
                *controller.default_launch(self.path, arm),
                "--run-dir",
                stage,
                *argv,
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=controller.ROOT,
            check=False,
            timeout=180,
        )
        return proc


class TestEndToEnd(Release):
    def test_complete_mock_attempt_through_the_real_path(self):
        m = self.write(self.manifest())
        code = controller.run(m, self.runs, manifest_path=self.path)
        self.assertEqual(code, 0)
        report = controller.analyze(m, self.runs)
        self.assertTrue(report["complete"], report["problems"])
        self.assertEqual(report["counts"]["valid"], 2)
        for arm in m["arms"]:
            cell = controller.cell_dir(self.runs, arm, 42, TASK)
            prov = json.loads(pathlib.Path(cell, "provenance.json").read_text())
            transcript_path = next(pathlib.Path(cell).rglob("transcript.json"))
            transcript = json.loads(transcript_path.read_text())
            sent = transcript["steps"][0]["content"]  # recorded by the engine
            self.assertEqual(sha(sent), prov["system_prompt_sha256"])
            self.assertEqual(
                prov["system_prompt_sha256"], m["arms"][arm]["system_prompt_sha256"]
            )
            self.assertEqual(NEW_SENTENCE in sent, arm == "verification")
            self.assertEqual(OLD_SENTENCE in sent, arm == "control")
            self.assertGreaterEqual(prov["model_calls"], 1)
            self.assertEqual(prov["source_verification"], "git+filesystem")
            self.assertEqual(prov["benchmark_argv"], controller.bench_argv(m, 42, TASK))
            self.assertTrue(os.path.exists(os.path.join(cell, "attempt.json")))
        # a second run finds nothing to do and re-runs nothing
        self.assertEqual(controller.run(m, self.runs, manifest_path=self.path), 0)

    def test_prompt_pin_that_does_not_match_the_prompt_is_refused_before_inference(
        self,
    ):
        m = self.manifest()
        m["arms"]["control"]["system_prompt_sha256"] = "0" * 64
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED, proc.stderr)
        self.assertIn("would receive differs", proc.stderr)
        self.assertEqual(
            sorted(os.listdir(self.stage)), ["launch-refused.json"]
        )  # no engine run

    def test_edited_prompt_content_that_differs_from_its_pin_is_refused(self):
        m = self.manifest()
        m["arms"]["verification"]["prompt_edit"]["replace"][1] = (
            NEW_SENTENCE + " Extra."
        )
        proc = self.launch(m, "verification")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("pinned hash", proc.stderr)
        self.assertNotIn("config.json", os.listdir(self.stage))

    def test_prompt_edit_that_does_not_apply_exactly_once_is_refused(self):
        m = self.manifest()
        m["arms"]["verification"]["prompt_edit"]["replace"][0] = "no such sentence"
        with self.assertRaises(ValueError):
            launcher.prompt_hashes(m, "verification")  # registration fails loudly
        proc = self.launch(m, "verification")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("exactly once", proc.stderr)

    def test_wrong_model_fingerprint_is_refused_before_inference(self):
        m = self.manifest()
        self.fake_ollama(self.home, self.MODEL, weights="other-weights")
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("weights_digest differs", proc.stderr)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

    def test_source_drift_in_the_manifest_pin_is_refused_before_inference(self):
        m = self.manifest()
        m["benchmark"]["source_sha256"] = "0" * 64
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("registered digest", proc.stderr)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

    def test_altered_bench_arguments_are_refused(self):
        m = self.manifest()
        proc = self.launch(m, "control", "--steps", "99")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

    def test_attempt_of_another_arm_or_seed_is_refused(self):
        m = self.manifest()
        self.write(m)
        stage = os.path.join(self.tmp, "stage-seed")
        os.makedirs(stage)
        argv = controller.bench_argv(m, 99, TASK)  # seed 99 is not in the matrix
        proc = subprocess.run(
            [
                *controller.default_launch(self.path, "control"),
                "--run-dir",
                stage,
                *argv,
            ],
            capture_output=True,
            text=True,
            cwd=controller.ROOT,
            check=False,
        )
        self.assertEqual(proc.returncode, launcher.REFUSED)

    def test_shadowed_engine_module_is_detected(self):
        m = self.manifest()
        argv = controller.bench_argv(m, 42, TASK)
        clean, _ = launcher.pre_inference_problems(m, "control", argv)
        self.assertEqual(clean, [])
        fake = types.SimpleNamespace(__file__="/somewhere/else/classifier.py")
        with mock.patch.dict(sys.modules, {"bench.classifier": fake}):
            shadowed, _ = launcher.pre_inference_problems(m, "control", argv)
        self.assertTrue(any("bench.classifier was loaded from" in p for p in shadowed))

    def test_frozen_manifest_cannot_be_launched(self):
        m = self.manifest()
        m["frozen"] = True
        m["integrity"] = "experiments/exp2_records/SHA256SUMS"
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertEqual(os.listdir(self.stage), [])

    def test_verify_command_passes_the_release_and_fails_a_drifted_pin(self):
        self.write(self.manifest())
        self.assertEqual(controller.main(["verify", self.path]), 0)
        m = self.manifest()
        m["benchmark"]["source_sha256"] = "0" * 64
        self.write(m)
        self.assertEqual(controller.main(["verify", self.path]), 2)

    def test_controller_preflight_refuses_a_drifted_release_before_launching(self):
        m = self.manifest()
        m["benchmark"]["source_sha256"] = "0" * 64
        m = self.write(m)
        self.assertEqual(controller.run(m, self.runs, manifest_path=self.path), 2)
        self.assertFalse(os.path.exists(os.path.join(self.runs, "control")))

    def test_refused_launch_leaves_evidence_and_blocks_a_silent_retry(self):
        m = self.manifest()
        m["arms"]["control"]["system_prompt_sha256"] = "0" * 64
        m = self.write(m)
        self.assertEqual(controller.run(m, self.runs, manifest_path=self.path), 3)
        # the quarantined attempt keeps the refusal reason
        kept = controller.cell_dir(self.runs, "control", 42, TASK) + ".invalid-1"
        self.assertTrue(os.path.exists(os.path.join(kept, "launch-refused.json")))
        self.assertEqual(controller.run(m, self.runs, manifest_path=self.path), 2)


class TestIncompleteAndObserved(Release):
    def run_with_fake_engine(self, fake_main, arm: str = "control"):
        m = self.write(self.manifest())
        stage = os.path.join(self.tmp, f"stage-{arm}")
        os.makedirs(stage)
        argv = ["--manifest", self.path, "--arm", arm, "--run-dir", stage]
        argv += controller.bench_argv(m, 42, TASK)
        with mock.patch.object(engine, "main", fake_main):
            code = launcher.main(argv)
        return m, stage, code

    def request(self, content: str):
        msgs = [
            {"role": "system", "content": content},
            {"role": "user", "content": "x"},
        ]
        return engine._mock_chat_stream("m", msgs, 1, 0)

    def test_engine_crash_keeps_the_attempt_and_writes_no_provenance(self):
        def crash():
            raise RuntimeError("boom")

        m, stage, code = self.run_with_fake_engine(crash)
        self.assertEqual(code, launcher.FAILED)
        files = os.listdir(stage)
        self.assertIn("launch-started.json", files)
        self.assertIn("launch-failed.json", files)
        self.assertNotIn("provenance.json", files)
        found = controller.validate_attempt(m, "control", 42, TASK, stage)
        self.assertTrue(found)  # the controller would quarantine it

    def test_engine_that_never_calls_the_model_is_not_accepted(self):
        _m, stage, code = self.run_with_fake_engine(lambda: 0)
        self.assertEqual(code, launcher.FAILED)
        self.assertNotIn("provenance.json", os.listdir(stage))
        self.assertIn(
            "no model call", pathlib.Path(stage, "launch-failed.json").read_text()
        )

    def test_a_different_prompt_at_the_model_call_is_detected(self):
        def wrong_prompt():
            self.request("some other system prompt")
            return 0

        _m, stage, code = self.run_with_fake_engine(wrong_prompt)
        self.assertEqual(code, launcher.FAILED)
        self.assertNotIn("provenance.json", os.listdir(stage))
        self.assertIn(
            "differs from the pinned",
            pathlib.Path(stage, "launch-failed.json").read_text(),
        )

    def test_a_mid_run_prompt_change_is_detected(self):
        def two_prompts():
            sent = engine.SYSTEM_PROMPT_TOOLS.format(max_steps=20)
            self.request(sent)
            self.request(sent + " (changed)")
            return 0

        _m, stage, code = self.run_with_fake_engine(two_prompts)
        self.assertEqual(code, launcher.FAILED)
        self.assertNotIn("provenance.json", os.listdir(stage))

    def test_clean_twin_with_the_pinned_prompt_writes_provenance(self):
        def honest():
            self.request(engine.SYSTEM_PROMPT_TOOLS.format(max_steps=20))
            return 1  # a failed task exits non-zero by design

        m, stage, code = self.run_with_fake_engine(honest)
        self.assertEqual(code, 1)
        prov = json.loads(pathlib.Path(stage, "provenance.json").read_text())
        self.assertEqual(
            prov["system_prompt_sha256"], m["arms"]["control"]["system_prompt_sha256"]
        )
        self.assertEqual(prov["model_calls"], 1)

    def test_the_engine_prompt_is_the_arms_prompt(self):
        seen = {}

        def capture():
            seen["prompt"] = engine.SYSTEM_PROMPT_TOOLS
            self.request(seen["prompt"].format(max_steps=20))
            return 0

        self.run_with_fake_engine(capture, arm="verification")
        self.assertIn(NEW_SENTENCE, seen["prompt"])
        self.assertNotIn(OLD_SENTENCE, seen["prompt"])


class TestManifestIdentityFields(Quiet):
    def runnable(self) -> dict[str, Any]:
        m = controller.load_manifest(os.path.join(HERE, "manifests", "exp2.json"))
        m = copy.deepcopy(m)
        for key in ("integrity", "aborted", "provenance_file"):
            m.pop(key, None)
        m["frozen"] = False
        m["benchmark"].update(
            commit="0" * 40, source_files=SOURCE_FILES, source_sha256="1" * 64
        )
        for arm in m["arms"].values():
            arm["system_prompt_sha256"] = "2" * 64
        return m

    def test_clean_runnable_manifest_is_valid(self):
        self.assertEqual(controller.validate_manifest(self.runnable()), [])

    def test_each_identity_defect_is_rejected(self):
        def mutate(change):
            m = self.runnable()
            change(m)
            return controller.validate_manifest(m)

        defects = {
            "no source digest": lambda m: m["benchmark"].pop("source_sha256"),
            "no source files": lambda m: m["benchmark"].pop("source_files"),
            "short commit": lambda m: m["benchmark"].update(commit="c027a89"),
            "engine not listed": lambda m: m["benchmark"].update(
                source_files=["bench/x.py"]
            ),
            "path traversal": lambda m: m["benchmark"].update(
                source_files=[*SOURCE_FILES, "../x"]
            ),
            "absolute path": lambda m: m["benchmark"].update(
                source_files=[*SOURCE_FILES, "/etc/passwd"]
            ),
            "bad digest": lambda m: m["benchmark"].update(source_sha256="xyz"),
            "arm without observed-prompt pin": lambda m: m["arms"]["control"].pop(
                "system_prompt_sha256"
            ),
            "bad prompt edit": lambda m: m["arms"]["control"].update(
                prompt_edit={"x": 1}
            ),
            "empty replacement": lambda m: m["arms"]["control"].update(
                prompt_edit={"replace": ["", "y"]}
            ),
        }
        for label, change in defects.items():
            self.assertTrue(mutate(change), label)

    def test_frozen_manifests_need_no_identity(self):
        m = controller.load_manifest(os.path.join(HERE, "manifests", "exp2.json"))
        self.assertEqual(controller.validate_manifest(m), [])


if __name__ == "__main__":
    unittest.main()
