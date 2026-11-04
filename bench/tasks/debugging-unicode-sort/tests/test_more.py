import unicodedata
import unittest

from names import sort_names


class TestSortNames(unittest.TestCase):
    def test_case_insensitive_primary(self):
        got = sort_names(["beta", "Alpha", "gamma", "Beta"])
        self.assertEqual([n.lower() for n in got],
                         ["alpha", "beta", "beta", "gamma"])

    def test_deterministic_tie_break(self):
        self.assertEqual(sort_names(["ab", "AB"]), ["AB", "ab"])
        self.assertEqual(sort_names(["AB", "ab"]), ["AB", "ab"])

    def test_accents_sort_with_base_letter(self):
        got = sort_names(["zebra", "Émile", "apple", "école"])
        self.assertEqual(got, ["apple", "école", "Émile", "zebra"])

    def test_composed_and_decomposed_compare_equal(self):
        composed = unicodedata.normalize("NFC", "é")
        decomposed = unicodedata.normalize("NFD", "é")
        got = sort_names([composed, decomposed])
        self.assertEqual(len(got), 2)
        self.assertEqual(set(got), {composed, decomposed})

    def test_non_ascii_does_not_raise(self):
        for names in (["日本語", "abc"], ["ß", "ss"], ["İstanbul", "istanbul"]):
            self.assertEqual(len(sort_names(names)), len(names))

    def test_input_not_mutated(self):
        data = ["b", "a"]
        sort_names(data)
        self.assertEqual(data, ["b", "a"])

    def test_empty(self):
        self.assertEqual(sort_names([]), [])

    def test_rejects_non_strings(self):
        for bad in ([1], [None], ["a", 2]):
            with self.assertRaises(TypeError):
                sort_names(bad)


if __name__ == "__main__":
    unittest.main()
