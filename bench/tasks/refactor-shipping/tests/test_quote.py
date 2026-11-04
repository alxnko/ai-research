import inspect
import unittest

import shipping
from shipping import quote, quote_apac, quote_eu, quote_us

CASES = [(0, False), (0, True), (1, False), (1, True), (2.5, False),
         (2.5, True), (10, True), (-1, False), (None, False), (0.001, True)]


class TestUnifiedQuote(unittest.TestCase):
    def test_matches_original_formula(self):
        for carrier in ("us", "eu", "apac"):
            r = shipping.RATES[carrier]
            for w, exp in CASES:
                if w is None or w < 0:
                    want = None
                else:
                    want = int(r["base"] + r["per_kg"] * w
                               + (r["express"] if exp else 0))
                self.assertEqual(quote(carrier, w, exp), want,
                                 f"{carrier} w={w} express={exp}")

    def test_entrypoints_delegate(self):
        calls = []
        original = shipping.quote
        shipping.quote = lambda c, w, e=False: calls.append((c, w, e)) or 1
        try:
            self.assertEqual(quote_us(1), 1)
            self.assertEqual(quote_eu(2, True), 1)
            self.assertEqual(quote_apac(3), 1)
        finally:
            shipping.quote = original
        self.assertEqual(calls, [("us", 1, False), ("eu", 2, True),
                                 ("apac", 3, False)])

    def test_entrypoints_match_quote(self):
        for w, exp in CASES:
            self.assertEqual(quote_us(w, exp), quote("us", w, exp))
            self.assertEqual(quote_eu(w, exp), quote("eu", w, exp))
            self.assertEqual(quote_apac(w, exp), quote("apac", w, exp))

    def test_unknown_carrier(self):
        self.assertIsNone(quote("mars", 1))

    def test_single_formula(self):
        src = inspect.getsource(shipping)
        self.assertEqual(src.count('rate["base"]'), 1)


if __name__ == "__main__":
    unittest.main()
