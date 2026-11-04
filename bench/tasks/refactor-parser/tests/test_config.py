import inspect
import unittest

import config
from config import ConfigError, parse_colon, parse_equals, parse_kv

SAMPLE_EQ = "a=1\n# comment\n\n  b = two words  \nc=x=y\n"
SAMPLE_CO = "a:1\n# comment\n\n  b : two words  \nc:x:y\n"


class TestParsers(unittest.TestCase):
    def test_equals(self):
        self.assertEqual(parse_equals(SAMPLE_EQ),
                         {"a": "1", "b": "two words", "c": "x=y"})

    def test_colon(self):
        self.assertEqual(parse_colon(SAMPLE_CO),
                         {"a": "1", "b": "two words", "c": "x:y"})

    def test_parse_kv_generic(self):
        self.assertEqual(parse_kv(SAMPLE_EQ, "="), parse_equals(SAMPLE_EQ))
        self.assertEqual(parse_kv(SAMPLE_CO, ":"), parse_colon(SAMPLE_CO))
        self.assertEqual(parse_kv("a|b", "|"), {"a": "b"})

    def test_empty_and_comments(self):
        self.assertEqual(parse_equals(""), {})
        self.assertEqual(parse_colon("\n\n# x\n"), {})

    def test_bad_line_raises(self):
        with self.assertRaises(ConfigError):
            parse_equals("novalue")
        with self.assertRaises(ConfigError):
            parse_colon("novalue")

    def test_first_separator_only(self):
        self.assertEqual(parse_equals("k=a=b=c"), {"k": "a=b=c"})
        self.assertEqual(parse_colon("k:a:b"), {"k": "a:b"})

    def test_delegation(self):
        calls = []
        original = config.parse_kv
        config.parse_kv = lambda t, s: calls.append(s) or {}
        try:
            parse_equals("a=1")
            parse_colon("a:1")
        finally:
            config.parse_kv = original
        self.assertEqual(calls, ["=", ":"])

    def test_single_implementation(self):
        src = inspect.getsource(config)
        self.assertEqual(src.count("for line in text.splitlines()"), 1)
        self.assertEqual(src.count("def parse_kv"), 1)


if __name__ == "__main__":
    unittest.main()
