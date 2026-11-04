import threading
import time
import unittest

from idem import run_once


class TestRunOnce(unittest.TestCase):
    def test_runs_once(self):
        store, calls = {}, []
        r1 = run_once(store, "k", lambda: calls.append(1) or "result")
        r2 = run_once(store, "k", lambda: calls.append(2) or "other")
        self.assertEqual(r1, "result")
        self.assertEqual(r2, "result")
        self.assertEqual(calls, [1])

    def test_distinct_keys_run_separately(self):
        store, calls = {}, []
        run_once(store, "a", lambda: calls.append("a"))
        run_once(store, "b", lambda: calls.append("b"))
        self.assertEqual(calls, ["a", "b"])

    def test_failure_is_not_cached(self):
        store = {}
        with self.assertRaises(ValueError):
            run_once(store, "k", lambda: (_ for _ in ()).throw(ValueError("boom")))
        self.assertNotIn("k", store)
        self.assertEqual(run_once(store, "k", lambda: "ok"), "ok")

    def test_result_objects_are_returned_identically(self):
        store, obj = {}, {"a": [1]}
        self.assertIs(run_once(store, "k", lambda: obj), obj)
        self.assertIs(run_once(store, "k", lambda: {}), obj)

    def test_threads_run_fn_exactly_once(self):
        store, calls = {}, []
        lock = threading.Lock()
        results, errors = [], []

        def slow():
            with lock:
                calls.append(1)
            time.sleep(0.2)
            return "v"

        def worker():
            try:
                results.append(run_once(store, "shared", slow))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, ["v"] * 8)

    def test_concurrent_failure_propagates_to_all(self):
        store = {}
        lock = threading.Lock()
        calls, errors = [], []

        def boom():
            with lock:
                calls.append(1)
            time.sleep(0.1)
            raise RuntimeError("nope")

        def worker():
            try:
                run_once(store, "k", boom)
            except RuntimeError as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(errors, ["nope"] * 4)


if __name__ == "__main__":
    unittest.main()
