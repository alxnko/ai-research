import collections
import threading
import time
import unittest

from idem import run_once

# A correct implementation never waits this long; the bound is only reached
# (and the test fails with a clear message) when an implementation blocks one
# call on another that the task says it must not wait for.
BLOCKED = 2.0  # a call that must not wait should finish in milliseconds
ENTER = 3.0  # a call that must start should start in milliseconds
LONG = 5.0  # cleanup joins; reached only after an assertion has already failed


class Gate:
    """A fn that reports entering, then blocks until released."""

    def __init__(self, result=None, error=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()
        self._result = result
        self._error = error

    def __call__(self):
        with self._lock:
            self.calls += 1
        self.entered.set()
        if not self.release.wait(LONG):
            raise AssertionError("gate was never released")
        if self._error is not None:
            raise self._error
        return self._result


def in_thread(fn, *args):
    """Run fn(*args) in a thread; return (thread, box) with box['v'] / box['e']."""
    box = {}

    def target():
        try:
            box["v"] = fn(*args)
        except BaseException as exc:
            box["e"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def await_entry(test, gate, workers):
    """Wait until fn is entered. If every caller already finished without
    entering fn (e.g. run_once raised), fail at once with that error."""
    deadline = time.monotonic() + ENTER
    while time.monotonic() < deadline:
        if gate.entered.wait(0.02):
            return
        if not any(thread.is_alive() for thread, _ in workers):
            break
    errors = [box["e"] for _, box in workers if "e" in box]
    test.fail(f"fn was never entered; run_once raised {errors[:1]!r}")


class TestSequential(unittest.TestCase):
    def test_runs_once(self):
        store, calls = {}, []
        r1 = run_once(store, "k", lambda: calls.append(1) or "result")
        r2 = run_once(store, "k", lambda: calls.append(2) or "other")
        self.assertEqual((r1, r2), ("result", "result"))
        self.assertEqual(calls, [1])

    def test_distinct_keys_run_separately(self):
        store, calls = {}, []
        run_once(store, "a", lambda: calls.append("a"))
        run_once(store, "b", lambda: calls.append("b"))
        self.assertEqual(calls, ["a", "b"])

    def test_failure_is_not_cached(self):
        store = {}

        def boom():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            run_once(store, "k", boom)
        self.assertNotIn("k", store)
        self.assertEqual(run_once(store, "k", lambda: "ok"), "ok")

    def test_result_objects_are_returned_identically(self):
        store, obj = {}, {"a": [1]}
        self.assertIs(run_once(store, "k", lambda: obj), obj)
        self.assertIs(run_once(store, "k", lambda: {}), obj)

    def test_falsy_results_are_cached(self):
        store, calls = {}, []
        for _ in range(3):
            self.assertIsNone(run_once(store, "k", lambda: calls.append(1)))
        self.assertEqual(calls, [1])

    def test_any_mutable_mapping_works_as_a_store(self):
        store, calls = collections.UserDict(), []
        run_once(store, "k", lambda: calls.append(1) or 7)
        self.assertEqual(run_once(store, "k", lambda: calls.append(2)), 7)
        self.assertEqual(calls, [1])


class TestSameKeyConcurrency(unittest.TestCase):
    def test_threads_run_fn_exactly_once(self):
        # Repeated rounds with a fresh key each time. The short sleep only
        # widens the window in which a racy implementation would start fn a
        # second time; a correct implementation passes however it is timed.
        for round_no in range(15):
            store, key = {}, f"k{round_no}"
            gate = Gate(result="v")
            started = [threading.Event() for _ in range(6)]

            def call(i, gate=gate, store=store, key=key, started=started):
                started[i].set()
                return run_once(store, key, gate)

            workers = [in_thread(call, i) for i in range(6)]
            await_entry(self, gate, workers)
            for event in started:
                self.assertTrue(event.wait(ENTER))
            time.sleep(0.01)
            gate.release.set()
            for thread, box in workers:
                thread.join(LONG)
                self.assertFalse(thread.is_alive())
                self.assertEqual(box, {"v": "v"})
            self.assertEqual(gate.calls, 1, f"round {round_no}")

    def test_concurrent_failure_reaches_every_waiter_once(self):
        store = {}
        error = RuntimeError("nope")
        gate = Gate(error=error)
        started = [threading.Event() for _ in range(4)]

        def call(i):
            started[i].set()
            return run_once(store, "k", gate)

        workers = [in_thread(call, i) for i in range(4)]
        await_entry(self, gate, workers)
        for event in started:
            self.assertTrue(event.wait(ENTER))
        time.sleep(0.01)
        gate.release.set()
        for thread, box in workers:
            thread.join(LONG)
            self.assertIs(box.get("e"), error)
        self.assertEqual(gate.calls, 1)
        self.assertNotIn("k", store)
        # nothing was cached and nothing is left in flight: a retry runs fn
        self.assertEqual(run_once(store, "k", lambda: "ok"), "ok")

    def test_new_caller_cannot_start_before_waiters_are_signaled(self):
        # The owner fails while a waiter is parked. A caller arriving right
        # after the failure may retry, but only once the failure has been
        # delivered; it must not run fn while the in-flight call is still
        # unresolved. (A retry that starts after the first call finished is
        # legitimate; one that starts well before it returned is not.)
        store, error = {}, RuntimeError("first")
        gate = Gate(error=error)
        owner = in_thread(run_once, store, "k", gate)
        await_entry(self, gate, [owner])
        waiter = in_thread(run_once, store, "k", lambda: "never")
        time.sleep(0.05)  # let the waiter park on the in-flight call
        ran_at = []

        def late_retry(delay):
            time.sleep(delay)
            while not ran_at:
                try:
                    return run_once(
                        store, "k", lambda: ran_at.append(time.monotonic()) or "retry"
                    )
                except RuntimeError:
                    continue  # joined the failing call: not a duplicate start

        gate.release.set()
        # staggered arrivals, so some land just after the failure begins
        late = [in_thread(late_retry, 0.02 * n) for n in range(6)]
        owner[0].join(LONG)
        returned = time.monotonic()
        waiter[0].join(LONG)
        for thread, _ in late:
            thread.join(LONG)
        self.assertIs(owner[1].get("e"), error)
        self.assertIs(waiter[1].get("e"), error)
        self.assertEqual(gate.calls, 1)
        if ran_at:
            # a correct implementation lets the retry start at most a few
            # milliseconds before the owner's raise is observed
            self.assertGreaterEqual(
                ran_at[0], returned - 0.1, "fn restarted before the failure was delivered"
            )

    def test_waiters_are_released_when_storing_the_result_fails(self):
        class Refusing(collections.UserDict):
            def __setitem__(self, key, value):
                raise OSError("disk full")

        store, gate = Refusing(), Gate(result="v")
        owner = in_thread(run_once, store, "k", gate)
        await_entry(self, gate, [owner])
        waiter = in_thread(run_once, store, "k", lambda: "never")
        time.sleep(0.05)
        gate.release.set()
        owner[0].join(BLOCKED)
        waiter[0].join(BLOCKED)
        self.assertFalse(waiter[0].is_alive(), "a waiter was left blocked")
        self.assertIsInstance(owner[1].get("e"), OSError)


class TestStoreScoping(unittest.TestCase):
    """The same key in different stores is an independent request."""

    def _held_call(self, store, key, gate):
        thread, box = in_thread(run_once, store, key, gate)
        await_entry(self, gate, [(thread, box)])
        return thread, box

    def _assert_second_store_is_independent(self, store_a, store_b):
        gate_a = Gate(result="first")
        thread_a, box_a = self._held_call(store_a, "same", gate_a)

        calls_b = []
        thread_b, box_b = in_thread(
            run_once, store_b, "same", lambda: calls_b.append(1) or "second"
        )
        thread_b.join(BLOCKED)
        blocked = thread_b.is_alive()
        gate_a.release.set()  # always release, so a failing run still ends
        thread_a.join(LONG)
        thread_b.join(LONG)
        self.assertFalse(blocked, "a call on store B waited for store A's call")

        self.assertEqual(box_b, {"v": "second"})
        self.assertEqual(box_a, {"v": "first"})
        self.assertEqual(calls_b, [1])
        self.assertEqual(gate_a.calls, 1)
        self.assertEqual(dict(store_a), {"same": "first"})
        self.assertEqual(dict(store_b), {"same": "second"})

    def test_equal_but_distinct_dict_stores(self):
        self._assert_second_store_is_independent({}, {})

    def test_non_dict_stores(self):
        self._assert_second_store_is_independent(
            collections.UserDict(), collections.UserDict()
        )

    def test_failure_in_one_store_does_not_reach_another(self):
        store_a, store_b = {}, {}
        error = RuntimeError("only store A")
        gate_a = Gate(error=error)
        thread_a, box_a = self._held_call(store_a, "same", gate_a)

        thread_b, box_b = in_thread(run_once, store_b, "same", lambda: "fine")
        thread_b.join(BLOCKED)
        blocked = thread_b.is_alive()
        gate_a.release.set()
        thread_a.join(LONG)
        thread_b.join(LONG)
        self.assertFalse(blocked, "a call on store B waited for store A's call")

        self.assertIs(box_a.get("e"), error)
        self.assertEqual(box_b, {"v": "fine"})
        self.assertNotIn("same", store_a)
        self.assertEqual(store_b, {"same": "fine"})

    def test_different_keys_in_one_store_do_not_wait_for_each_other(self):
        store = {}
        gate = Gate(result="slow")
        thread_a, box_a = self._held_call(store, "a", gate)

        thread_b, box_b = in_thread(run_once, store, "b", lambda: "quick")
        thread_b.join(BLOCKED)
        blocked = thread_b.is_alive()
        gate.release.set()
        thread_a.join(LONG)
        thread_b.join(LONG)
        self.assertFalse(blocked, "key 'b' waited for the in-flight call on key 'a'")
        self.assertEqual(box_b, {"v": "quick"})
        self.assertEqual(box_a, {"v": "slow"})
        self.assertEqual(store, {"a": "slow", "b": "quick"})

    def test_same_key_same_store_still_waits_while_other_store_runs(self):
        # contention on one (store, key) must still collapse to a single call
        store_a, store_b = {}, {}
        gate_a, gate_b = Gate(result="A"), Gate(result="B")
        thread_a1, box_a1 = self._held_call(store_a, "k", gate_a)
        thread_b1, box_b1 = self._held_call(store_b, "k", gate_b)
        extra = [in_thread(run_once, store_a, "k", lambda: "late") for _ in range(3)]
        time.sleep(0.01)
        gate_a.release.set()
        gate_b.release.set()
        for thread in (thread_a1, thread_b1, *(t for t, _ in extra)):
            thread.join(LONG)
        self.assertEqual(box_a1, {"v": "A"})
        self.assertEqual(box_b1, {"v": "B"})
        self.assertEqual(gate_a.calls, 1)
        self.assertEqual(gate_b.calls, 1)
        for _, box in extra:
            self.assertEqual(box, {"v": "A"})


if __name__ == "__main__":
    unittest.main()
