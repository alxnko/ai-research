import unittest

from theme import resolve

TOKENS = {
    "--blue": " #0055ff ",
    "--brand": "var(--blue)",
    "--brand-2": "var(--brand)",
    "--space": "4px 8px",
    "--combo": "1px solid var(--brand)",
    "--missing-fallback": "var(--nope, #fff)",
    "--nested": "var(--nope, var(--also-nope, red))",
    "--two": "var(--brand) var(--space)",
    "--cycle-a": "var(--cycle-b)",
    "--cycle-b": "var(--cycle-a)",
    "--self": "var(--self)",
}


class TestResolve(unittest.TestCase):
    def test_direct(self):
        self.assertEqual(resolve(TOKENS, "--blue"), "#0055ff")

    def test_chain(self):
        self.assertEqual(resolve(TOKENS, "--brand-2"), "#0055ff")

    def test_multiple_references(self):
        self.assertEqual(resolve(TOKENS, "--combo"), "1px solid #0055ff")
        self.assertEqual(resolve(TOKENS, "--two"), "#0055ff 4px 8px")

    def test_fallback_used_when_unknown(self):
        self.assertEqual(resolve(TOKENS, "--missing-fallback"), "#fff")

    def test_nested_fallback(self):
        self.assertEqual(resolve(TOKENS, "--nested"), "red")

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            resolve(TOKENS, "--nope")

    def test_unknown_without_fallback_raises(self):
        with self.assertRaises(KeyError):
            resolve({"--a": "var(--b)"}, "--a")

    def test_cycles_raise(self):
        for name in ("--cycle-a", "--cycle-b", "--self"):
            with self.assertRaises(ValueError) as cm:
                resolve(TOKENS, name)
            self.assertIn("cycle", str(cm.exception).lower())

    def test_whitespace_stripped(self):
        self.assertEqual(resolve({"--a": "   red   "}, "--a"), "red")

    def test_no_vars(self):
        self.assertEqual(resolve({"--a": "0 0 2px rgb(0 0 0 / 20%)"},
                                 "--a"), "0 0 2px rgb(0 0 0 / 20%)")

    def test_fallback_with_comma_value(self):
        tokens = {"--shadow": "var(--nope, 0 0 2px rgba(0,0,0,.2))"}
        self.assertEqual(resolve(tokens, "--shadow"), "0 0 2px rgba(0,0,0,.2)")

    def test_deep_chain(self):
        tokens = {"--a": "var(--b)", "--b": "var(--c)", "--c": "1px"}
        self.assertEqual(resolve(tokens, "--a"), "1px")


if __name__ == "__main__":
    unittest.main()
