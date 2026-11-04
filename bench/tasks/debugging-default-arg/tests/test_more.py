import unittest

from cache import SessionCache


class TestDefaults(unittest.TestCase):
    def test_no_shared_state(self):
        a, b = SessionCache(), SessionCache()
        a.add("one")
        self.assertEqual(b.snapshot(), [])
        b.add("two")
        self.assertEqual(a.snapshot(), ["one"])

    def test_explicit_history_is_used(self):
        c = SessionCache(["seed"])
        self.assertEqual(c.snapshot(), ["seed"])

    def test_explicit_history_is_copied(self):
        seed = ["seed"]
        c = SessionCache(seed)
        c.add("more")
        self.assertEqual(seed, ["seed"])

    def test_limit_keeps_newest(self):
        c = SessionCache(limit=3)
        for i in range(5):
            c.add(i)
        self.assertEqual(c.snapshot(), [2, 3, 4])

    def test_snapshot_isolation(self):
        c = SessionCache()
        c.add("x")
        snap = c.snapshot()
        snap.append("y")
        snap.clear()
        self.assertEqual(c.snapshot(), ["x"])

    def test_clear(self):
        c = SessionCache()
        c.add("x")
        c.clear()
        self.assertEqual(c.snapshot(), [])
        c.add("y")
        self.assertEqual(c.snapshot(), ["y"])

    def test_two_instances_independent_after_clear(self):
        a = SessionCache()
        b = SessionCache()
        a.add("x")
        a.clear()
        b.add("y")
        self.assertEqual(a.snapshot(), [])
        self.assertEqual(b.snapshot(), ["y"])


if __name__ == "__main__":
    unittest.main()
