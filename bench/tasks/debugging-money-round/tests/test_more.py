import unittest

from money import Money, apply_tax, multiply, total


class TestExactMoney(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(Money(0).cents, 0)
        self.assertEqual(Money("0.05").cents, 5)
        self.assertEqual(Money("1").cents, 100)
        self.assertEqual(Money("19.99").cents, 1999)
        self.assertEqual(Money("-0.05").cents, -5)

    def test_str(self):
        for cents, text in [(0, "0.00"), (5, "0.05"), (100, "1.00"),
                            (1234, "12.34"), (-5, "-0.05"), (-1234, "-12.34")]:
            self.assertEqual(str(Money(cents)), text)

    def test_arithmetic(self):
        self.assertEqual(Money(100) + Money(1), Money(101))
        self.assertEqual(Money(100) - Money(1), Money(99))
        self.assertLess(Money(1), Money(2))

    def test_total_exact_not_float(self):
        self.assertEqual(total([Money("0.10")] * 3), 30)
        self.assertEqual(total([]), 0)
        self.assertEqual(total([Money(1)] * 1000), 1000)

    def test_multiply(self):
        self.assertEqual(multiply(199, 3), 597)
        self.assertEqual(multiply(5, 0), 0)

    def test_tax_rounding(self):
        self.assertEqual(apply_tax(1010, 5), 51)     # 50.5 up
        self.assertEqual(apply_tax(1009, 5), 50)     # 50.45 down
        self.assertEqual(apply_tax(1000, 0), 0)
        self.assertEqual(apply_tax(1000, 100), 1000)
        self.assertEqual(apply_tax(333, 7), 23)      # 23.31

    def test_no_float_arithmetic(self):
        import inspect
        src = inspect.getsource(__import__("money"))
        self.assertNotIn("/ 100", src.replace("/ 100)", "/ 100)"))


if __name__ == "__main__":
    unittest.main()
