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
import importlib
import importlib.util
import io
import json
import os
import pathlib
import py_compile
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
        self.assertTrue(any("not an annotated tag" in p for p in self.problems(m)))
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

    def test_symlinked_fixture_directory_is_detected(self):
        # os.walk lists a symlinked directory without entering it, so a content
        # digest cannot see it, yet the engine's copytree follows it
        target = os.path.join(self.tmp, "extra_tests")
        os.makedirs(target)
        pathlib.Path(target, "test_hidden.py").write_text("FAIL = 1\n")
        os.symlink(target, os.path.join(self.root, "bench/tasks/t1/tests/extra"))
        found = self.problems()
        self.assertTrue(any("tests/extra is a symlink" in p for p in found), found)
        self.assertEqual(self.problems(use_git=False), found[:1])  # offline too

    def test_symlinked_fixture_file_is_detected(self):
        link = os.path.join(self.root, "bench/tasks/t1/repo/b.py")
        os.symlink(os.path.join(self.root, "bench/tasks/t1/repo/a.py"), link)
        self.assertTrue(any("b.py is a symlink" in p for p in self.problems()))

    def test_empty_fixture_directory_is_detected(self):
        os.makedirs(os.path.join(self.root, "bench/tasks/t1/repo/empty"))
        self.assertTrue(any("empty directory" in p for p in self.problems()))

    def test_executable_bit_change_is_detected_against_the_tag(self):
        os.chmod(os.path.join(self.root, "bench/tasks/t1/repo/a.py"), 0o755)
        found = self.problems()
        self.assertTrue(any("mode differs from the tag" in p for p in found), found)
        self.assertEqual(self.problems(use_git=False), [])  # the tag is the reference

    def test_git_replace_cannot_swap_the_tagged_objects(self):
        blob = git(
            self.root,
            "rev-parse",
            f"{self.manifest['benchmark']['commit']}:bench/classifier.py",
        )
        evil = os.path.join(self.tmp, "evil.py")
        pathlib.Path(evil).write_text("CLASSIFIER = 'evil'\n")
        evil_blob = git(self.root, "hash-object", "-w", evil)
        git(self.root, "replace", blob, evil_blob)
        self.put("bench/classifier.py", "CLASSIFIER = 'evil'\n")
        found = self.problems(self.reregister())
        self.assertTrue(
            any("classifier.py differs from the tag" in p for p in found), found
        )

    def test_lightweight_tag_is_not_a_release(self):
        git(self.root, "tag", "-d", "vT")
        git(self.root, "tag", "vT")  # lightweight
        self.assertTrue(any("not an annotated tag" in p for p in self.problems()))

    def test_missing_git_is_a_refusal_not_a_crash(self):
        with mock.patch.dict(os.environ, {"PATH": ""}):
            found = self.problems()
        self.assertTrue(any("annotated tag" in p for p in found), found)

    def test_registered_source_set_may_not_be_smaller_than_the_engine(self):
        for missing in (
            "bench/telemetry.py",
            "bench/classifier.py",
            "bench/__init__.py",
        ):
            m = copy.deepcopy(self.manifest)
            m["benchmark"]["source_files"] = [f for f in SOURCE_FILES if f != missing]
            self.assertTrue(any("must include" in p for p in self.problems(m)), missing)

    def test_dir_digest_matches_the_engine_algorithm(self):
        task = os.path.join(controller.ROOT, "bench", "tasks", TASK)
        self.assertEqual(source_identity.dir_digest(task), engine.task_digest(TASK))


def engine_modules() -> dict[str, Any]:
    names = ("bench", "bench.bench", "bench.classifier", "bench.telemetry")
    return {k: v for k, v in sys.modules.items() if k in names}


def have_release() -> bool:
    return source_identity.resolve_tag(controller.ROOT, TAG) is not None


class Release(Quiet):
    """A runnable manifest against the real v0.5 release, a fake Ollama store."""

    MODEL = "qwen3.5:4b-q4_K_M"

    def setUp(self) -> None:
        super().setUp()
        if not have_release():
            if os.environ.get("CI"):  # a shallow checkout must not pass vacuously
                self.fail(f"tag {TAG} not fetched: set fetch-depth: 0 in CI")
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
        # Tests that call the launcher in-process run inside a process that has
        # imported other bench modules (selfcheck); the subprocess tests exercise
        # the unfiltered check. Explicit `modules` arguments still win.
        real_check = launcher.loaded_module_problems
        self.enterContext(
            mock.patch.object(
                launcher,
                "loaded_module_problems",
                lambda m, modules=None: real_check(
                    m, engine_modules() if modules is None else modules
                ),
            )
        )
        self.enterContext(launcher.isolated_bytecode())
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
        m["inference"] = "mock"  # these tests run the engine's mock model
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
        approved = controller.manifest_sha256(self.write(m))
        stage = os.path.join(self.tmp, f"stage-{arm}-{len(os.listdir(self.tmp))}")
        os.makedirs(stage)
        self.stage = stage
        argv = controller.bench_argv(m, 42, TASK)
        proc = subprocess.run(
            [
                *controller.default_launch(self.path, arm, approved),
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
        approved = controller.manifest_sha256(self.write(m))
        stage = os.path.join(self.tmp, "stage-seed")
        os.makedirs(stage)
        argv = controller.bench_argv(m, 99, TASK)  # seed 99 is not in the matrix
        proc = subprocess.run(
            [
                *controller.default_launch(self.path, "control", approved),
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
        clean, _ = launcher.pre_inference_problems(m, "control", argv, engine_modules())
        self.assertEqual(clean, [])
        fake = types.SimpleNamespace(__file__="/somewhere/else/classifier.py")
        shadow = {**engine_modules(), "bench.classifier": fake}
        shadowed, _ = launcher.pre_inference_problems(m, "control", argv, shadow)
        self.assertTrue(any("bench.classifier was loaded from" in p for p in shadowed))

    def test_mock_model_leaking_into_a_real_experiment_is_refused(self):
        m = self.manifest()
        m["inference"] = "ollama"  # a real experiment; BENCH_MOCK_MODEL=1 is exported
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("BENCH_MOCK_MODEL", proc.stderr)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

    def test_mock_experiment_without_the_mock_switch_is_refused(self):
        m = self.manifest()  # registers inference "mock"
        with mock.patch.dict(os.environ):
            os.environ.pop("BENCH_MOCK_MODEL")
            proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

    def test_manifest_that_is_not_the_approved_one_is_refused(self):
        m = self.manifest()
        self.write(m)
        stage = os.path.join(self.tmp, "stage-hash")
        os.makedirs(stage)
        cmd = controller.default_launch(self.path, "control", "0" * 64)
        argv = controller.bench_argv(m, 42, TASK)
        proc = subprocess.run(
            [*cmd, "--run-dir", stage, *argv],
            capture_output=True,
            text=True,
            cwd=controller.ROOT,
            check=False,
        )
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("not the one the controller approved", proc.stderr)

    def test_a_manifest_cannot_name_its_own_launcher(self):
        m = self.manifest()
        m["arms"]["control"]["launch"] = [sys.executable, "-c", "pass"]
        self.assertTrue(any("launch" in p for p in controller.validate_manifest(m)))

    def test_retry_requests_go_through_the_observed_call(self):
        """Full engine run, text actuator: the first reply is empty (thinking
        only), so the engine issues its retry request. Every call is observed."""
        from bench.telemetry import StreamResult  # noqa: PLC0415

        m = self.manifest()
        m["settings"]["actuator"] = "text"
        m["arms"] = {"control": {}, "verification": {}}  # same prompt in both
        for arm in m["arms"]:
            m["arms"][arm].update(launcher.prompt_hashes(m, arm))
        self.write(m)
        real, calls = engine._mock_chat_stream, []

        def scripted(model, msgs, *rest, **kwargs):
            calls.append([x["role"] for x in msgs])
            res = real(model, msgs, *rest, **kwargs)
            if len(calls) == 1:  # empty content, some thinking: triggers the retry
                return StreamResult(
                    content="", thinking="hmm", tool_calls=[], telemetry=res.telemetry
                )
            return res

        stage = os.path.join(self.tmp, "stage-retry")
        os.makedirs(stage)
        argv = [
            "--manifest",
            self.path,
            "--manifest-sha256",
            controller.manifest_sha256(controller.load_manifest(self.path)),
        ]
        argv += [
            "--arm",
            "control",
            "--run-dir",
            stage,
            *controller.bench_argv(m, 42, TASK),
        ]
        with mock.patch.object(engine, "_mock_chat_stream", scripted):
            launcher.main(argv)
        prov = json.loads(pathlib.Path(stage, "provenance.json").read_text())
        self.assertGreaterEqual(len(calls), 2)  # the empty reply and its retry
        self.assertTrue(
            any("assistant" in roles for roles in calls), calls
        )  # the retry
        self.assertEqual(prov["model_calls"], len(calls))
        self.assertEqual(
            prov["system_prompt_sha256"], m["arms"]["control"]["system_prompt_sha256"]
        )

    def test_source_is_verified_again_after_the_engine_is_imported(self):
        m = self.manifest()
        self.write(m)
        stage = os.path.join(self.tmp, "stage-recheck")
        os.makedirs(stage)
        argv = ["--manifest", self.path, "--arm", "control", "--run-dir", stage]
        argv += [
            "--manifest-sha256",
            controller.manifest_sha256(controller.load_manifest(self.path)),
        ]
        argv += controller.bench_argv(m, 42, TASK)
        # clean before the import, drifted by the time the engine is loaded
        with mock.patch.object(
            launcher.source_identity,
            "verify_source",
            side_effect=[[], ["engine file changed during startup"]],
        ):
            self.assertEqual(launcher.main(argv), launcher.REFUSED)
        self.assertEqual(os.listdir(stage), ["launch-refused.json"])

    def test_drifted_source_is_refused_before_the_engine_is_imported(self):
        m = self.manifest()
        m["benchmark"]["source_sha256"] = "0" * 64
        approved = controller.manifest_sha256(self.write(m))
        stage = os.path.join(self.tmp, "stage-early")
        os.makedirs(stage)
        argv = [
            "--manifest",
            self.path,
            "--manifest-sha256",
            approved,
            "--arm",
            "control",
        ]
        argv += ["--run-dir", stage, *controller.bench_argv(m, 42, TASK)]
        code = (
            "import sys, experiments.launcher as l; "
            f"rc = l.main({argv!r}); print(rc, 'bench.bench' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=controller.ROOT,
            check=True,
        ).stdout.split()
        self.assertEqual(out, [str(launcher.REFUSED), "False"])

    def test_frozen_manifest_cannot_be_launched(self):
        m = self.manifest()
        m["frozen"] = True
        m["integrity"] = "experiments/exp2_records/SHA256SUMS"
        proc = self.launch(m, "control")
        self.assertEqual(proc.returncode, launcher.REFUSED)
        self.assertIn("runnable", proc.stderr)
        self.assertEqual(os.listdir(self.stage), ["launch-refused.json"])

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
        argv += ["--manifest-sha256", controller.manifest_sha256(m)]
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

    def test_sys_argv_is_restored_after_the_engine_runs(self):
        before = list(sys.argv)
        self.run_with_fake_engine(lambda: 0)
        self.assertEqual(sys.argv, before)

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

    def test_the_real_model_call_function_is_observed(self):
        """`chat_stream` (the real Ollama path) is wrapped like the mock one."""
        reached = []

        def stub(model, msgs, *rest, **kwargs):  # stands in for the HTTP call
            reached.append(msgs[0]["content"])

        def honest():
            engine.chat_stream(
                "m",
                [
                    {
                        "role": "system",
                        "content": engine.SYSTEM_PROMPT_TOOLS.format(max_steps=20),
                    },
                    {"role": "user", "content": "x"},
                ],
                1,
                0,
            )
            return 0

        with mock.patch.object(engine, "chat_stream", stub):
            _m, stage, code = self.run_with_fake_engine(honest)
        self.assertEqual(code, 0)
        self.assertEqual(len(reached), 1)  # the call went through to the function
        prov = json.loads(pathlib.Path(stage, "provenance.json").read_text())
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


class TestRecorder(Quiet):
    def test_every_request_is_observed_including_a_retry(self):
        recorder = launcher.Recorder()
        seen = []
        call = recorder.wrap(lambda *a, **k: seen.append(1))
        system = {"role": "system", "content": "S"}
        step = [system, {"role": "user", "content": "task"}]
        retry = [
            *step,
            {"role": "assistant", "content": "hmm"},
            {"role": "user", "content": "act"},
        ]
        call("m", step, 1, 0)  # positional, as the step loop calls it
        call("m", retry, 1, 0)  # the empty-reply retry
        call("m", msgs=step)  # keyword form
        self.assertEqual((recorder.calls, len(seen)), (3, 3))
        self.assertEqual(recorder.system_hashes, {sha("S")})
        self.assertEqual(launcher.observed_problems(recorder, sha("S")), [])

    def test_a_changed_system_prompt_on_any_call_invalidates(self):
        recorder = launcher.Recorder()
        call = recorder.wrap(lambda *a, **k: None)
        user = {"role": "user", "content": "x"}
        call("m", [{"role": "system", "content": "S"}, user], 1, 0)
        call("m", [{"role": "system", "content": "S2"}, user, user], 1, 0)  # a retry
        self.assertEqual(len(recorder.system_hashes), 2)
        self.assertTrue(launcher.observed_problems(recorder, sha("S")))

    def test_a_request_without_a_system_message_never_matches(self):
        recorder = launcher.Recorder()
        recorder.wrap(lambda *a, **k: None)(
            "m", [{"role": "user", "content": "S"}], 1, 0
        )
        self.assertTrue(launcher.observed_problems(recorder, sha("S")))

    def test_no_calls_is_a_problem(self):
        self.assertTrue(launcher.observed_problems(launcher.Recorder(), sha("S")))


class TestBytecodeIsolation(Quiet):
    """A planted .pyc must never replace the verified source."""

    def build(self) -> str:
        root = os.path.join(self.tmp, "tree")
        for rel in ("experiments", "bench"):
            os.makedirs(os.path.join(root, rel))
            for name in os.listdir(os.path.join(controller.ROOT, rel)):
                if name.endswith(".py"):
                    shutil.copy(
                        os.path.join(controller.ROOT, rel, name),
                        os.path.join(root, rel),
                    )
        target = os.path.join(root, "bench", "classifier.py")
        evil = os.path.join(self.tmp, "evil.py")
        pathlib.Path(evil).write_text(pathlib.Path(target).read_text() + "\nEVIL = 1\n")
        with mock.patch.object(sys, "pycache_prefix", None):  # the in-tree location
            cached = importlib.util.cache_from_source(target)
        py_compile.compile(
            evil,
            cfile=cached,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        return root

    def probe(self, root: str, first: str) -> str:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}
        code = f"{first}import bench.classifier as c; print(hasattr(c, 'EVIL'))"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    def test_forged_bytecode_loads_normally_but_not_under_the_launcher(self):
        root = self.build()
        self.assertEqual(self.probe(root, ""), "True")  # the attack works: sensitivity
        # keep a reference: a collected context manager would restore at once
        isolated = (
            "import experiments.launcher as l; cm = l.isolated_bytecode(); "
            "cm.__enter__(); "
        )
        self.assertEqual(self.probe(root, isolated), "False")

    def test_unisolated_bytecode_cache_is_a_problem(self):
        m = {"benchmark": {"source_files": SOURCE_FILES}}
        with launcher.isolated_bytecode():
            self.assertEqual(launcher.loaded_module_problems(m, engine_modules()), [])
        with mock.patch.object(sys, "pycache_prefix", None):
            found = launcher.loaded_module_problems(m, engine_modules())
        self.assertTrue(any("not isolated" in p for p in found), found)
        saved = sys.pycache_prefix
        with launcher.isolated_bytecode():
            pass
        self.assertEqual(sys.pycache_prefix, saved)  # scoped: restored afterwards

    def test_unregistered_bench_module_is_a_problem(self):
        m = {"benchmark": {"source_files": SOURCE_FILES}}
        extra = types.SimpleNamespace(
            __file__=os.path.join(controller.ROOT, "bench", "selfcheck.py")
        )
        found = launcher.loaded_module_problems(
            m, {**engine_modules(), "bench.selfcheck": extra}
        )
        self.assertTrue(any("bench.selfcheck" in p for p in found), found)


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
            "unknown inference backend": lambda m: m.update(inference="gpu"),
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
