"""Behavioural oracle for exact money.

Expected values come from independent definitions (Fraction and plain integer
arithmetic), never from the code under test, and never from inspecting its
source. Generated inputs use a fixed seed so every run is identical. Large
values (beyond 2**53) are where any binary-float shortcut gives itself away.
"""

import math
import random
import unittest
from fractions import Fraction

from money import Money, apply_tax, multiply, total

BIG = 2**53
RNG = random.Random(20261109)


def oracle_tax(cents, rate):
    """floor(cents * rate / 100 + 1/2): round half up, exactly."""
    exact = Fraction(cents) * Fraction(str(rate)) / 100
    return math.floor(exact + Fraction(1, 2))


def oracle_text(cents):
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"


class Mismatches:
    """Collect the first few wrong answers and report them in one assertion,
    so a wrong implementation fails fast and readably instead of emitting
    thousands of tracebacks."""

    def __init__(self, limit=5):
        self.limit = limit
        self.rows = []

    def check(self, label, call, want, exact_type=None):
        try:
            got = call()
        except Exception as exc:
            got = f"raised {type(exc).__name__}: {exc}"
        wrong_type = exact_type is not None and type(got) is not exact_type
        if (got != want or wrong_type) and len(self.rows) < self.limit:
            self.rows.append((label, got, want))

    def assert_none(self, test):
        test.assertEqual(self.rows, [], "first mismatches: (input, got, want)")


def amounts(n=200):
    fixed = [0, 1, 5, 99, 100, 101, 1234, -5, -1234, BIG - 1, BIG, BIG + 1]
    fixed += [-(BIG + 1), 10**20 + 7, -(10**20) - 7]
    return fixed + [RNG.randrange(-(10**22), 10**22) for _ in range(n)]


class TestParse(unittest.TestCase):
    def test_decimal_strings(self):
        bad = Mismatches()
        for text, cents in [
            ("0", 0),
            ("1", 100),
            ("0.05", 5),
            ("0.5", 50),
            ("12.3", 1230),
            ("12.34", 1234),
            ("19.99", 1999),
            ("-0.05", -5),
            ("-12.34", -1234),
            ("90071992547409.93", 9007199254740993),
            ("-90071992547409.93", -9007199254740993),
        ]:
            bad.check(text, lambda t=text: Money(t).cents, cents)
        bad.assert_none(self)

    def test_integer_cents(self):
        bad = Mismatches()
        for cents in amounts(50):
            bad.check(cents, lambda c=cents: Money(c).cents, cents)
        bad.assert_none(self)

    def test_parse_inverts_format_for_any_amount(self):
        bad = Mismatches()
        for cents in amounts():
            bad.check(cents, lambda c=cents: Money(oracle_text(c)).cents, cents)
        bad.assert_none(self)


class TestFormat(unittest.TestCase):
    def test_known_values(self):
        for cents, text in [
            (0, "0.00"),
            (5, "0.05"),
            (100, "1.00"),
            (1234, "12.34"),
            (-5, "-0.05"),
            (-1234, "-12.34"),
        ]:
            self.assertEqual(str(Money(cents)), text, cents)

    def test_matches_integer_formatting_for_any_amount(self):
        bad = Mismatches()
        for cents in amounts():
            bad.check(cents, lambda c=cents: str(Money(c)), oracle_text(cents))
        bad.assert_none(self)


class TestArithmetic(unittest.TestCase):
    def test_add_sub_compare_match_integers(self):
        values = amounts(60)
        bad = Mismatches()
        for a in values[:30]:
            for b in values[30:60]:
                x, y = Money(a), Money(b)
                bad.check((a, b, "add"), lambda: (x + y).cents, a + b)
                bad.check((a, b, "sub"), lambda: (x - y).cents, a - b)
                bad.check((a, b, "eq"), lambda: x == y, a == b)
                bad.check((a, b, "lt"), lambda: x < y, a < b)
        bad.assert_none(self)
        self.assertIsInstance(Money(1) + Money(2), Money)
        self.assertIsInstance(Money(1) - Money(2), Money)

    def test_equal_values_compare_equal(self):
        self.assertEqual(Money(100), Money("1.00"))
        self.assertNotEqual(Money(100), Money(101))
        self.assertFalse(Money(5) < Money(5))
        self.assertLess(Money(-1), Money(0))


class TestTotalAndMultiply(unittest.TestCase):
    def test_total_is_exact_integer_cents(self):
        self.assertEqual(total([]), 0)
        self.assertEqual(total([Money("0.10")] * 3), 30)
        self.assertEqual(total([Money(1)] * 1000), 1000)
        bad = Mismatches()
        for _ in range(60):
            cents = [RNG.randrange(-(10**20), 10**20) for _ in range(RNG.randint(1, 8))]
            bad.check(
                cents, lambda c=cents: total([Money(v) for v in c]), sum(cents), int
            )
        bad.assert_none(self)

    def test_total_beyond_float_precision(self):
        self.assertEqual(total([Money(BIG + 1), Money(1)]), BIG + 2)
        self.assertEqual(total([Money(BIG + 1), Money(BIG + 1)]), 2 * BIG + 2)

    def test_multiply_is_exact(self):
        self.assertEqual(multiply(199, 3), 597)
        self.assertEqual(multiply(5, 0), 0)
        self.assertEqual(multiply(0, 7), 0)
        bad = Mismatches()
        for _ in range(100):
            cents, qty = RNG.randrange(0, 10**15), RNG.randrange(0, 10**6)
            bad.check((cents, qty), lambda c=cents, q=qty: multiply(c, q), cents * qty, int)
        bad.assert_none(self)
        self.assertEqual(multiply(BIG + 1, 3), 3 * BIG + 3)


class TestApplyTax(unittest.TestCase):
    def test_worked_examples(self):
        bad = Mismatches()
        for cents, rate, want in [
            (1010, 5, 51),  # 50.5 rounds up
            (1009, 5, 50),  # 50.45 rounds down
            (1011, 5, 51),  # 50.55 rounds up
            (1000, 0, 0),
            (1000, 100, 1000),
            (333, 7, 23),  # 23.31
            (0, 5, 0),
            (1, 50, 1),  # 0.5 -> 1
            (1, 49, 0),  # 0.49 -> 0
            (3, 50, 2),  # 1.5 -> 2
            (25, 10, 3),  # 2.5 -> 3 (not banker's 2)
            (35, 10, 4),  # 3.5 -> 4
            (200, "7.5", 15),  # exactly 15
            (10, "7.5", 1),  # 0.75 -> 1
            (6, "7.5", 0),  # 0.45 -> 0
            (1000, "0.5", 5),
            (1, "0.5", 0),  # 0.005 -> 0
        ]:
            bad.check((cents, rate), lambda c=cents, r=rate: apply_tax(c, r), want)
        bad.assert_none(self)

    def test_every_exact_half_boundary_rounds_up(self):
        # Search for amounts whose tax is exactly x.5 and check both sides.
        bad = Mismatches()
        for rate in (5, 10, 15, 25, 50, "2.5", "12.5", "0.5", "33.5", "7.5"):
            found = 0
            for cents in range(0, 4000):
                exact = Fraction(cents) * Fraction(str(rate)) / 100
                if exact - math.floor(exact) == Fraction(1, 2):
                    found += 1
                    for c in (cents - 1, cents, cents + 1):
                        if c >= 0:
                            bad.check(
                                (c, rate),
                                lambda c=c, r=rate: apply_tax(c, r),
                                oracle_tax(c, rate),
                            )
            self.assertGreater(found, 0, f"no boundary found for rate {rate}")
        bad.assert_none(self)

    def test_generated_amounts_and_rates_match_the_oracle(self):
        rates = [0, 1, 5, 7, 8, 19, 20, 25, 100, "2.5", "7.5", "12.5", "33.5"]
        bad = Mismatches()
        for _ in range(600):
            rate = RNG.choice(rates)
            # a string rate can only be multiplied as text by careless code
            # (cents * "7.5" repeats a string), so keep those amounts small
            cents = RNG.randrange(0, 10**5 if isinstance(rate, str) else 10**9)
            bad.check(
                (cents, rate),
                lambda c=cents, r=rate: apply_tax(c, r),
                oracle_tax(cents, rate),
                int,
            )
        bad.assert_none(self)

    def test_amounts_beyond_float_precision(self):
        bad = Mismatches()
        for cents, rate in [
            (BIG + 1, 100),
            (BIG + 1, 50),
            (BIG + 3, 50),
            (10**17 + 5, 50),
            (10**18 + 1, 5),
            (10**20 + 7, "7.5"),
            (3 * BIG + 1, 100),
        ]:
            bad.check(
                (cents, rate),
                lambda c=cents, r=rate: apply_tax(c, r),
                oracle_tax(cents, rate),
            )
        bad.assert_none(self)

    def test_large_generated_amounts_match_the_oracle(self):
        bad = Mismatches()
        for _ in range(200):
            cents = RNG.randrange(BIG, 10**20)
            rate = RNG.choice([5, 7, 19, 100, "2.5", "7.5", "33.5"])
            bad.check(
                (cents, rate),
                lambda c=cents, r=rate: apply_tax(c, r),
                oracle_tax(cents, rate),
            )
        bad.assert_none(self)


if __name__ == "__main__":
    unittest.main()
