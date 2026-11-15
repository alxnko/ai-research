"""Offline tests of the server-side model identity checks.

A fake Ollama (stdlib HTTP server on an ephemeral loopback port) serves
/api/tags and /api/ps from mutable state, so every mismatch the contract has to
refuse is simulated without a GPU or a real server.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest
from typing import Any

from bench import bench
from experiments import launcher, server_identity

TAG = "qwen3.5:4b-q4_K_M"
HEX = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
MANIFEST_DIGEST = f"sha256:{HEX}"  # the manifest's prefixed form
OTHER = "5" * 64
CTX = 8192


def tags_entry(name: str = TAG, digest: Any = HEX) -> dict[str, Any]:
    return {"name": name, "model": name, "digest": digest, "size": 1}


def ps_entry(
    name: str = TAG, digest: Any = HEX, ctx: Any = CTX, **extra: Any
) -> dict[str, Any]:
    entry = {
        "name": name,
        "model": name,
        "digest": digest,
        "size": 3_000_000_000,
        "size_vram": 2_900_000_000,
    }
    if ctx is not None:
        entry["context_length"] = ctx
    return {**entry, **extra}


class FakeOllama:
    """/api/tags and /api/ps from state a test can change at any time. A value
    that is a str is sent verbatim (malformed bodies); an int is an HTTP status."""

    def __init__(self) -> None:
        self.tags: Any = {"models": [tags_entry(), tags_entry("other:1b", OTHER)]}
        self.ps: Any = {"models": [ps_entry()]}
        self.hits: list[str] = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                fake.hits.append(self.path)
                body = {"/api/tags": fake.tags, "/api/ps": fake.ps}.get(self.path)
                status = body if isinstance(body, int) else 200
                data = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server: Any = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        ).start()

    def close(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


class Served(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)

    def advertised(self) -> list[str]:
        return server_identity.advertised(TAG, MANIFEST_DIGEST, self.fake.base)[0]

    def loaded(self, ctx: int = CTX) -> list[str]:
        return server_identity.loaded(TAG, MANIFEST_DIGEST, ctx, self.fake.base)[0]


class TestDigests(unittest.TestCase):
    def test_prefixed_and_bare_forms_agree(self):
        norm = server_identity.normalise_digest
        self.assertEqual(norm(HEX), MANIFEST_DIGEST)
        self.assertEqual(norm(MANIFEST_DIGEST), MANIFEST_DIGEST)
        self.assertEqual(norm(HEX.upper()), MANIFEST_DIGEST)

    def test_anything_that_is_not_a_full_digest_is_rejected(self):
        for bad in (None, 5, "", "d8b0f5e9760c", HEX[:-1], HEX + "0", "sha512:" + HEX):
            self.assertIsNone(server_identity.normalise_digest(bad), bad)


class TestAdvertised(Served):
    def test_the_registered_model_is_accepted(self):
        self.assertEqual(self.advertised(), [])
        found = server_identity.advertised(TAG, MANIFEST_DIGEST, self.fake.base)[1]
        self.assertEqual(found, {"digest": MANIFEST_DIGEST})

    def test_every_mismatch_is_a_problem(self):
        cases = {
            "model not served": {"models": [tags_entry("other:1b", OTHER)]},
            "no models at all": {"models": []},
            "wrong digest under the name": {"models": [tags_entry(digest=OTHER)]},
            "abbreviated digest": {"models": [tags_entry(digest=HEX[:12])]},
            "digest missing": {"models": [{"name": TAG, "model": TAG}]},
            "digest not a string": {"models": [tags_entry(digest=7)]},
            "listed twice": {"models": [tags_entry(), tags_entry()]},
            "listed twice with another digest": {
                "models": [tags_entry(), tags_entry(digest=OTHER)]
            },
            "only a similar name": {"models": [tags_entry("qwen3.5:4b", HEX)]},
            "models is not a list": {"models": {"a": 1}},
            "models missing": {},
            "entry is not an object": {"models": ["x"]},
            "not a JSON object": "[]",
            "not JSON": "<html>",
            "empty body": "",
            "server error": 500,
        }
        for label, state in cases.items():
            self.fake.tags = state
            self.assertTrue(self.advertised(), f"{label} was accepted")

    def test_an_unreachable_server_is_a_problem_never_a_fallback(self):
        self.fake.close()
        found = server_identity.advertised(TAG, MANIFEST_DIGEST, self.fake.base)
        self.assertTrue(found[0])
        self.assertEqual(found[1], {})

    def test_a_tag_without_a_version_means_latest(self):
        self.fake.tags = {"models": [tags_entry("tiny:latest")]}
        self.assertEqual(
            server_identity.advertised("tiny", MANIFEST_DIGEST, self.fake.base)[0], []
        )


class TestLoaded(Served):
    def test_the_loaded_registered_model_is_accepted(self):
        self.assertEqual(self.loaded(), [])
        seen = server_identity.loaded(TAG, MANIFEST_DIGEST, CTX, self.fake.base)[1]
        self.assertEqual(seen["digest"], MANIFEST_DIGEST)
        self.assertEqual(seen["context_length"], CTX)
        self.assertEqual(seen["size_vram"], 2_900_000_000)

    def test_every_mismatch_is_a_problem(self):
        cases = {
            "nothing loaded": {"models": []},
            "a different model is loaded": {"models": [ps_entry("other:1b", OTHER)]},
            "same name, different weights": {"models": [ps_entry(digest=OTHER)]},
            "digest missing": {"models": [ps_entry(digest=None)]},
            "context not reported": {"models": [ps_entry(ctx=None)]},
            "context as text": {"models": [ps_entry(ctx="8192")]},
            "context as bool": {"models": [ps_entry(ctx=True)]},
            "different context": {"models": [ps_entry(ctx=4096)]},
            "loaded twice": {"models": [ps_entry(), ps_entry()]},
            "malformed": "not json",
            "server error": 503,
        }
        for label, state in cases.items():
            self.fake.ps = state
            self.assertTrue(self.loaded(), f"{label} was accepted")

    def test_another_loaded_model_beside_ours_is_not_an_identity_problem(self):
        self.fake.ps = {"models": [ps_entry("other:1b", OTHER), ps_entry()]}
        self.assertEqual(self.loaded(), [])


class FakeEngine:
    @staticmethod
    def chat_stream(model, msgs, ctx, ngpu, url="http://127.0.0.1:11434/api/chat"):
        pass


class TestSameInstance(unittest.TestCase):
    def test_engine_and_check_agree_on_the_default_address(self):
        self.assertEqual(server_identity.instance_problems(FakeEngine), [])

    def test_a_different_inspected_instance_is_refused(self):
        found = server_identity.instance_problems(FakeEngine, "http://127.0.0.1:11435")
        self.assertTrue(found)
        self.assertIn("11434", found[0])

    def test_a_different_host_is_refused(self):
        found = server_identity.instance_problems(FakeEngine, "http://localhost:11434")
        self.assertTrue(found)

    def test_an_engine_whose_target_cannot_be_read_is_refused(self):
        class NoUrl:
            @staticmethod
            def chat_stream(model, msgs):
                pass

        class NotCallable:
            chat_stream = 5

        for engine in (NoUrl, NotCallable, object()):
            self.assertTrue(server_identity.instance_problems(engine), engine)

    def test_the_frozen_engine_really_targets_the_checked_address(self):
        self.assertEqual(server_identity.instance_problems(bench), [])


class TestWatcher(Served):
    def watcher(self) -> server_identity.Watcher:
        return server_identity.Watcher(TAG, MANIFEST_DIGEST, CTX, self.fake.base)

    def test_a_consistent_run_is_accepted_and_recorded(self):
        w = self.watcher()
        self.assertEqual(w.begin(), [])
        w.observe()
        w.observe()  # a retry is a second call
        self.assertEqual(w.finish(), [])
        rec = w.record()
        self.assertEqual(rec["loaded_checks"], 2)
        self.assertEqual(rec["advertised_digest"], MANIFEST_DIGEST)
        self.assertEqual(rec["loaded_digest"], MANIFEST_DIGEST)
        self.assertEqual(rec["loaded_context"], CTX)
        self.assertEqual(rec["instance"], self.fake.base)
        self.assertEqual(rec["attestation"], server_identity.ATTESTATION)

    def test_a_refusal_before_inference_names_the_stage(self):
        self.fake.tags = {"models": []}
        found = self.watcher().begin()
        self.assertTrue(found)
        self.assertTrue(all(p.startswith("before inference") for p in found))

    def test_the_model_is_never_loaded(self):
        w = self.watcher()
        w.begin()
        w.observe()  # a call that did not load the model
        self.fake.ps = {"models": []}
        w.observe()
        self.assertTrue(w.finish())

    def test_the_model_is_gone_or_replaced_at_completion(self):
        for state in ({"models": []}, {"models": [ps_entry(digest=OTHER)]}):
            w = self.watcher()
            w.begin()
            w.observe()
            self.assertEqual(w.problems, [])  # every call saw the right model
            self.fake.ps = state
            found = w.finish()
            self.assertTrue(any("at completion" in p for p in found), found)
            self.fake.ps = {"models": [ps_entry()]}

    def test_the_model_is_replaced_while_the_attempt_runs(self):
        w = self.watcher()
        w.begin()
        w.observe()
        self.fake.ps = {"models": [ps_entry(digest=OTHER)]}  # swapped mid-run
        w.observe()
        found = w.finish()
        self.assertTrue(any("after model call 2" in p for p in found), found)

    def test_a_tag_repulled_after_the_last_call_fails_at_completion(self):
        w = self.watcher()
        w.begin()
        w.observe()
        self.fake.tags = {"models": [tags_entry(digest=OTHER)]}
        found = w.finish()
        self.assertTrue(any("at completion" in p for p in found), found)

    def test_a_server_that_disappears_during_the_run_fails_it(self):
        w = self.watcher()
        w.begin()
        w.observe()
        self.fake.tags = self.fake.ps = 500
        self.assertTrue(w.finish())

    def test_a_context_change_during_the_run_fails_it(self):
        w = self.watcher()
        w.begin()
        w.observe()
        self.fake.ps = {"models": [ps_entry(ctx=4096)]}
        w.observe()
        self.assertTrue(w.finish())

    def test_no_observation_at_all_is_a_problem(self):
        w = self.watcher()
        w.begin()
        self.assertTrue(w.finish())  # a run that never called the model


class TestRecorderObserver(unittest.TestCase):
    def test_the_observer_runs_after_every_request_even_a_failed_one(self):
        seen: list[str] = []
        recorder = launcher.Recorder(lambda: seen.append("observed"))
        order: list[str] = []

        def call(*args: Any, **kwargs: Any) -> None:
            order.append("call")
            if len(order) == 2:
                raise TimeoutError("server stalled")

        wrapped = recorder.wrap(call)
        msgs = [{"role": "system", "content": "S"}]
        wrapped("m", msgs, 1, 0)
        with self.assertRaises(TimeoutError):
            wrapped("m", msgs, 1, 0)
        self.assertEqual(seen, ["observed", "observed"])

    def test_no_observer_means_no_server_calls(self):
        recorder = launcher.Recorder()
        recorder.wrap(lambda *a, **k: None)("m", [{"role": "system", "content": ""}])
        self.assertEqual(recorder.calls, 1)


if __name__ == "__main__":
    unittest.main()
