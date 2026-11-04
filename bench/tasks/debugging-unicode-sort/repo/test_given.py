import unittest
from names import sort_names


class TestSort(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertEqual(sort_names(["beta", "Alpha", "gamma"]),
                         ["Alpha", "beta", "gamma"])

    def test_accented_names(self):
        self.assertEqual(sort_names(["Zoe", "Émile", "anna"]),
                         ["anna", "Émile", "Zoe"])


if __name__ == "__main__":
    unittest.main()
