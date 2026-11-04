import unittest
from cache import SessionCache


class TestCache(unittest.TestCase):
    def test_separate_sessions_start_empty(self):
        a = SessionCache()
        a.add("hello")
        b = SessionCache()
        self.assertEqual(b.snapshot(), [])

    def test_snapshot_is_a_copy(self):
        c = SessionCache()
        c.add("x")
        snap = c.snapshot()
        snap.append("y")
        self.assertEqual(c.snapshot(), ["x"])


if __name__ == "__main__":
    unittest.main()
