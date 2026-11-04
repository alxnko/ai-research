import inspect
import unittest

import validators
from validators import (ValidationError, validate_bio, validate_name,
                        validate_username)

FIELDS = [
    ("name", validate_name, 1, 80),
    ("username", validate_username, 3, 20),
    ("bio", validate_bio, 0, 280),
]


class TestBehaviourPreserved(unittest.TestCase):
    def test_type_errors(self):
        for field, fn, _, _ in FIELDS:
            with self.assertRaises(ValidationError) as cm:
                fn(None)
            self.assertEqual(str(cm.exception), f"{field} must be text")

    def test_length_errors(self):
        for field, fn, low, high in FIELDS:
            with self.assertRaises(ValidationError) as cm:
                fn("x" * (high + 1))
            self.assertEqual(str(cm.exception),
                             f"{field} length must be {low}-{high}")
            if low > 0:
                with self.assertRaises(ValidationError) as cm:
                    fn("x" * (low - 1) if low > 1 else "")
                self.assertEqual(str(cm.exception),
                                 f"{field} length must be {low}-{high}")

    def test_success_strips(self):
        self.assertEqual(validate_name("  Ada  "), "Ada")
        self.assertEqual(validate_username("  ada  "), "ada")
        self.assertEqual(validate_bio("  hi  "), "hi")
        self.assertEqual(validate_bio(""), "")
        self.assertEqual(validate_bio("   "), "")

    def test_boundaries(self):
        self.assertEqual(validate_name("x"), "x")
        self.assertEqual(validate_name("x" * 80), "x" * 80)
        self.assertEqual(validate_username("abc"), "abc")
        self.assertEqual(validate_username("x" * 20), "x" * 20)
        self.assertEqual(len(validate_bio("x" * 280)), 280)

    def test_non_strings_rejected(self):
        for _, fn, _, _ in FIELDS:
            for bad in (123, None, ["x"], b"x"):
                with self.assertRaises(ValidationError):
                    fn(bad)


class TestRefactor(unittest.TestCase):
    def test_shared_helper_exists(self):
        self.assertTrue(hasattr(validators, "_check_text"))

    def test_all_validators_delegate(self):
        calls = []
        original = validators._check_text
        validators._check_text = lambda v, f, lo, hi: calls.append(f) or "ok"
        try:
            self.assertEqual(validate_name("x"), "ok")
            self.assertEqual(validate_username("x"), "ok")
            self.assertEqual(validate_bio("x"), "ok")
        finally:
            validators._check_text = original
        self.assertEqual(calls, ["name", "username", "bio"])

    def test_single_type_check(self):
        src = inspect.getsource(validators)
        self.assertEqual(src.count("isinstance(value, str)"), 1)


if __name__ == "__main__":
    unittest.main()
