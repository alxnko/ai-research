import unittest
from money import Money, apply_tax, total


class TestMoney(unittest.TestCase):
    def test_string_parse(self):
        self.assertEqual(Money("12.34").cents, 1234)

    def test_str_format(self):
        self.assertEqual(str(Money(5)), "0.05")

    def test_negative_format(self):
        self.assertEqual(str(Money(-5)), "-0.05")

    def test_total_is_exact(self):
        self.assertEqual(total([Money("0.10"), Money("0.20")]), 30)

    def test_tax_rounds_half_up(self):
        self.assertEqual(apply_tax(1010, 5), 51)


if __name__ == "__main__":
    unittest.main()
