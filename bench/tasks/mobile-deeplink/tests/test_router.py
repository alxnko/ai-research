import unittest

from router import Router

ROUTES = [
    ("/", "home"),
    ("/orders/:id", "order"),
    ("/orders/:id/items/:item", "item"),
    ("/files/*", "file"),
    ("/users/me", "me"),
]


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.r = Router(ROUTES)

    def test_static_route(self):
        self.assertEqual(self.r.match("/"), {"name": "home", "params": {}})

    def test_single_param(self):
        self.assertEqual(self.r.match("/orders/42"),
                         {"name": "order", "params": {"id": "42"}})

    def test_two_params(self):
        self.assertEqual(self.r.match("/orders/42/items/7"),
                         {"name": "item", "params": {"id": "42", "item": "7"}})

    def test_static_beats_param(self):
        self.assertEqual(self.r.match("/users/me"),
                         {"name": "me", "params": {}})

    def test_wildcard(self):
        self.assertEqual(self.r.match("/files/report.pdf"),
                         {"name": "file", "params": {"*": "report.pdf"}})

    def test_no_match(self):
        for path in ("/nope", "/orders/1/items", "/files", "/files/a/b"):
            self.assertIsNone(self.r.match(path))

    def test_segment_count_must_match(self):
        self.assertIsNone(self.r.match("/orders"))
        self.assertIsNone(self.r.match("/orders/1/extra"))

    def test_trailing_slash_ignored(self):
        self.assertEqual(self.r.match("/orders/42/")["name"], "order")
        self.assertEqual(self.r.match("/")["name"], "home")
        self.assertEqual(self.r.match("")["name"], "home")

    def test_query_and_fragment_ignored(self):
        self.assertEqual(self.r.match("/orders/42?ref=mail"),
                         {"name": "order", "params": {"id": "42"}})

    def test_percent_decoded(self):
        self.assertEqual(self.r.match("/orders/a%20b")["params"]["id"], "a b")

    def test_first_match_wins(self):
        r = Router([("/x/:a", "first"), ("/x/:b", "second")])
        self.assertEqual(r.match("/x/1")["name"], "first")

    def test_bad_patterns(self):
        with self.assertRaises(ValueError):
            Router([("orders", "nope")])
        with self.assertRaises(ValueError):
            Router([("/a", "one"), ("/a", "two")])


if __name__ == "__main__":
    unittest.main()
