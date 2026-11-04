import unittest

from ratelimit import RateLimiter


class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.rl = RateLimiter(limit=3, window=60)

    def test_allows_up_to_limit(self):
        self.assertEqual([self.rl.allow("k", 100 + i) for i in range(3)],
                         [True, True, True])
        self.assertFalse(self.rl.allow("k", 103))

    def test_rejects_record_nothing(self):
        for i in range(3):
            self.rl.allow("k", 100 + i)
        self.assertFalse(self.rl.allow("k", 103))
        self.assertEqual(self.rl.remaining("k", 103), 0)

    def test_window_slides(self):
        for i in range(3):
            self.rl.allow("k", 100 + i)
        self.assertFalse(self.rl.allow("k", 159))
        self.assertTrue(self.rl.allow("k", 160))  # 100 is now exactly 60 old

    def test_boundary_event_expires(self):
        self.rl.allow("k", 100)
        self.assertEqual(self.rl.remaining("k", 159), 2)
        self.assertEqual(self.rl.remaining("k", 160), 3)

    def test_keys_are_independent(self):
        for i in range(3):
            self.rl.allow("a", 100 + i)
        self.assertTrue(self.rl.allow("b", 103))
        self.assertFalse(self.rl.allow("a", 104))

    def test_remaining_counts_down(self):
        self.assertEqual(self.rl.remaining("k", 0), 3)
        self.rl.allow("k", 0)
        self.assertEqual(self.rl.remaining("k", 0), 2)

    def test_reset_at(self):
        self.assertIsNone(self.rl.reset_at("k", 0))
        self.rl.allow("k", 100)
        self.rl.allow("k", 110)
        self.assertEqual(self.rl.reset_at("k", 120), 160)
        self.assertEqual(self.rl.reset_at("k", 160), 170)

    def test_idle_state_is_dropped(self):
        self.rl.allow("k", 0)
        self.rl.remaining("k", 10_000)
        self.assertNotIn("k", self.rl._events)


if __name__ == "__main__":
    unittest.main()
