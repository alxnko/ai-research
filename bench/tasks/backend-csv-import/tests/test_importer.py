import unittest

from importer import import_users

CSV = """email,name,plan
ada@example.com,  Ada Lovelace  ,pro
grace@example.com,Grace Hopper,free
,No Email,free
bad-email,Bad,free
dup@example.com,First,free
DUP@example.com,Second,free
noname@example.com,   ,free
"""


class TestImport(unittest.TestCase):
    def setUp(self):
        self.out = import_users(CSV)

    def test_accepted_rows(self):
        self.assertEqual(self.out["accepted"], [
            {"email": "ada@example.com", "name": "Ada Lovelace"},
            {"email": "grace@example.com", "name": "Grace Hopper"},
            {"email": "dup@example.com", "name": "First"},
        ])

    def test_rejected_rows_and_reasons(self):
        self.assertEqual(self.out["rejected"], [
            {"row": 3, "reason": "missing_email"},
            {"row": 4, "reason": "bad_email"},
            {"row": 6, "reason": "duplicate"},
            {"row": 7, "reason": "missing_name"},
        ])

    def test_column_order_independent(self):
        out = import_users("name,email\nAda,ada@example.com\n")
        self.assertEqual(out["accepted"],
                         [{"email": "ada@example.com", "name": "Ada"}])

    def test_extra_columns_ignored(self):
        out = import_users("email,name,extra\na@b.co,A,x\n")
        self.assertEqual(out["accepted"], [{"email": "a@b.co", "name": "A"}])

    def test_blank_lines_skipped_and_not_counted(self):
        out = import_users("email,name\n\nbad,E\n\n,empty\n")
        self.assertEqual(out["rejected"], [
            {"row": 1, "reason": "bad_email"},
            {"row": 2, "reason": "missing_email"},
        ])

    def test_header_validation(self):
        for bad in ("", "name,plan\nA,pro\n", "email\n"):
            with self.assertRaises(ValueError):
                import_users(bad)

    def test_short_rows(self):
        out = import_users("email,name\na@b.co\n")
        self.assertEqual(out["rejected"], [{"row": 1, "reason": "missing_name"}])

    def test_email_validation(self):
        out = import_users("email,name\n"
                           "a@b.co,ok\n"
                           "a@b,no-dot\n"
                           "@b.co,no-local\n"
                           "a@b@c.co,two-at\n"
                           "A@B.CO,upper\n")
        self.assertEqual([a["email"] for a in out["accepted"]], ["a@b.co"])
        self.assertEqual([r["reason"] for r in out["rejected"]],
                         ["bad_email", "bad_email", "bad_email", "duplicate"])

    def test_quoted_fields_with_commas(self):
        out = import_users('email,name\n"a@b.co","Lovelace, Ada"\n')
        self.assertEqual(out["accepted"],
                         [{"email": "a@b.co", "name": "Lovelace, Ada"}])


if __name__ == "__main__":
    unittest.main()
