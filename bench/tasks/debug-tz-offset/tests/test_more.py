import unittest
from timeutil import format_offset, to_local_iso


class TestOffsets(unittest.TestCase):
    def test_signs_both_ways(self):
        self.assertEqual(format_offset(60), "+01:00")
        self.assertEqual(format_offset(-60), "-01:00")
        self.assertEqual(format_offset(-330), "-05:30")

    def test_sub_hour(self):
        self.assertEqual(format_offset(30), "+00:30")
        self.assertEqual(format_offset(-30), "-00:30")

    def test_zero(self):
        self.assertEqual(format_offset(0), "Z")

    def test_iso_negative_offset(self):
        self.assertEqual(to_local_iso(0, -300), "1969-12-31T19:00:00-05:00")

    def test_iso_zero(self):
        self.assertEqual(to_local_iso(86400, 0), "1970-01-02T00:00:00Z")

    def test_iso_india(self):
        self.assertEqual(to_local_iso(86400, 330), "1970-01-02T05:30:00+05:30")

    def test_large_offset(self):
        self.assertEqual(format_offset(840), "+14:00")
