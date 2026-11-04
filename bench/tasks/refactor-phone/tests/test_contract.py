import unittest
import contacts
from contacts import normalize_home, normalize_work, normalize_phone, format_contact

CASES = [
    ("5551234567", "+15551234567"),
    ("15551234567", "+15551234567"),
    ("+1 (555) 123-4567", "+15551234567"),
    ("555-123-4567", "+15551234567"),
    ("555123456", None),
    ("", None),
    ("abc", None),
    ("155512345678", None),
    (None, None),
]


class TestContract(unittest.TestCase):
    def test_normalize_phone_exists_and_matches(self):
        for raw, expected in CASES:
            got = normalize_phone(raw) if raw is not None else None
            self.assertEqual(got, expected, f"normalize_phone({raw!r})")

    def test_old_entrypoints_preserved(self):
        for raw, expected in CASES:
            if raw is None:
                continue
            self.assertEqual(normalize_home(raw), expected)
            self.assertEqual(normalize_work(raw), expected)

    def test_delegation(self):
        calls = []
        original = contacts.normalize_phone
        contacts.normalize_phone = lambda v: calls.append(v) or original(v)
        try:
            normalize_home("5551234567")
            normalize_work("5551234567")
        finally:
            contacts.normalize_phone = original
        self.assertEqual(len(calls), 2, "old entrypoints must delegate")

    def test_single_implementation(self):
        import inspect
        src = inspect.getsource(contacts)
        self.assertEqual(src.count("def normalize_phone"), 1)

    def test_format_contact(self):
        r = format_contact({"name": " Ada ", "home": "5551234567",
                            "work": "555-987-6543"})
        self.assertEqual(r, {"name": "Ada", "home": "+15551234567",
                             "work": "+15559876543"})
