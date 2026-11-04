import unittest
from api import paginate

ITEMS = list(range(1, 24))  # 23 items


class TestPaginate(unittest.TestCase):
    def test_basic_first_page(self):
        r = paginate(ITEMS, 1, 10)
        self.assertEqual(r["items"], list(range(1, 11)))
        self.assertEqual(r["total"], 23)
        self.assertEqual(r["pages"], 3)
        self.assertFalse(r["has_prev"])
        self.assertTrue(r["has_next"])

    def test_last_page_partial(self):
        r = paginate(ITEMS, 3, 10)
        self.assertEqual(r["items"], [21, 22, 23])
        self.assertFalse(r["has_next"])
        self.assertTrue(r["has_prev"])

    def test_page_beyond_end(self):
        r = paginate(ITEMS, 99, 10)
        self.assertEqual(r["items"], [])
        self.assertFalse(r["has_next"])
        self.assertTrue(r["has_prev"])

    def test_empty_collection(self):
        r = paginate([], 1, 10)
        self.assertEqual(r["items"], [])
        self.assertEqual(r["pages"], 0)
        self.assertFalse(r["has_next"])
        self.assertFalse(r["has_prev"])

    def test_exact_division(self):
        r = paginate(list(range(20)), 2, 10)
        self.assertEqual(r["pages"], 2)
        self.assertFalse(r["has_next"])

    def test_invalid_page(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError) as cm:
                paginate(ITEMS, bad, 10)
            self.assertIn("page", str(cm.exception))

    def test_invalid_per_page(self):
        for bad in (0, -5, 101):
            with self.assertRaises(ValueError) as cm:
                paginate(ITEMS, 1, bad)
            self.assertIn("per_page", str(cm.exception))

    def test_does_not_mutate(self):
        data = [[i] for i in range(5)]
        snapshot = [list(x) for x in data]
        paginate(data, 1, 2)
        self.assertEqual(data, snapshot)

    def test_accepts_any_sequence(self):
        r = paginate(tuple("abcde"), 2, 2)
        self.assertEqual(list(r["items"]), ["c", "d"])
