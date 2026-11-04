import unittest

from query import build, merge, parse


class TestParse(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse("a=1&b=2"), {"a": "1", "b": "2"})

    def test_leading_question_mark(self):
        self.assertEqual(parse("?a=1"), {"a": "1"})

    def test_repeated_keys_become_list(self):
        self.assertEqual(parse("t=a&t=b&t=c"), {"t": ["a", "b", "c"]})

    def test_key_without_value(self):
        self.assertEqual(parse("flag&a=1"), {"flag": "", "a": "1"})

    def test_plus_is_space(self):
        self.assertEqual(parse("q=hello+world"), {"q": "hello world"})

    def test_percent_decoding(self):
        self.assertEqual(parse("q=a%26b%3Dc"), {"q": "a&b=c"})

    def test_malformed_escape_kept(self):
        self.assertEqual(parse("q=100%"), {"q": "100%"})

    def test_empty(self):
        for value in ("", None, "?"):
            self.assertEqual(parse(value), {})

    def test_empty_pairs_ignored(self):
        self.assertEqual(parse("a=1&&b=2&"), {"a": "1", "b": "2"})


class TestBuild(unittest.TestCase):
    def test_sorted_stable_output(self):
        self.assertEqual(build({"b": "2", "a": "1"}), "a=1&b=2")

    def test_lists_repeat(self):
        self.assertEqual(build({"t": ["a", "b"]}), "t=a&t=b")

    def test_none_skipped(self):
        self.assertEqual(build({"a": None, "b": "x"}), "b=x")

    def test_encoding(self):
        self.assertEqual(build({"q": "a b&c"}), "q=a+b%26c")

    def test_round_trip(self):
        params = {"q": "hello world", "tags": ["x", "y"], "empty": ""}
        self.assertEqual(parse(build(params)), params)


class TestMerge(unittest.TestCase):
    def test_updates_move_to_end(self):
        self.assertEqual(merge("a=1&b=2", {"a": "9"}), "b=2&a=9")

    def test_adds_new_key(self):
        self.assertEqual(merge("a=1", {"z": "0"}), "a=1&z=0")

    def test_none_removes(self):
        self.assertEqual(merge("a=1&b=2", {"a": None}), "b=2")

    def test_list_value(self):
        self.assertEqual(merge("a=1", {"t": ["x", "y"]}), "a=1&t=x&t=y")

    def test_keeps_other_order(self):
        self.assertEqual(merge("z=1&a=2&m=3", {"new": "v"}), "z=1&a=2&m=3&new=v")


if __name__ == "__main__":
    unittest.main()
