import unittest
from timeutil import format_offset, to_local_iso


class TestTimeutil(unittest.TestCase):
    def test_offset_sign(self):
        self.assertEqual(format_offset(330), "+05:30")

    def test_offset_padding(self):
        self.assertEqual(format_offset(-90), "-01:30")

    def test_zero_offset(self):
        self.assertEqual(format_offset(0), "Z")

    def test_iso(self):
        self.assertEqual(to_local_iso(0, 330), "1970-01-01T05:30:00+05:30")


if __name__ == "__main__":
    unittest.main()
