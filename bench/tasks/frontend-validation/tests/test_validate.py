import unittest

from validate import clean, validate

GOOD = {"email": "A@B.co", "password": "abcdefgh12", "age": 30, "name": " Ada "}


def err(field, **over):
    fields = dict(GOOD)
    fields.update(over)
    return validate(fields).get(field)


class TestValidate(unittest.TestCase):
    def test_valid_input_has_no_errors(self):
        self.assertEqual(validate(GOOD), {})

    def test_email_rules(self):
        self.assertTrue(err("email", email=""))
        self.assertTrue(err("email", email="   "))
        self.assertTrue(err("email", email="abc.co"))
        self.assertTrue(err("email", email="a@bco"))
        self.assertTrue(err("email", email="@b.co"))
        self.assertTrue(err("email", email="a@b@c.co"))
        self.assertIsNone(err("email", email="  a@b.co  "))
        self.assertIsNone(err("email", email="user.name@sub.domain.org"))

    def test_password_rules(self):
        self.assertTrue(err("password", password=""))
        self.assertTrue(err("password", password="abc12"))
        self.assertTrue(err("password", password="abcdefghij"))
        self.assertTrue(err("password", password="1234567890"))
        self.assertIsNone(err("password", password="abcdefgh12"))

    def test_age_rules(self):
        self.assertTrue(err("age", age=12))
        self.assertTrue(err("age", age=121))
        self.assertTrue(err("age", age=30.5))
        self.assertTrue(err("age", age=30.0))
        self.assertTrue(err("age", age="30"))
        self.assertTrue(err("age", age=True))
        self.assertTrue(err("age", age=None))
        self.assertIsNone(err("age", age=13))
        self.assertIsNone(err("age", age=120))

    def test_name_rules(self):
        self.assertTrue(err("name", name="x" * 81))
        self.assertIsNone(err("name", name=""))
        self.assertIsNone(err("name", name="   "))
        self.assertIsNone(err("name", name="x" * 80))
        self.assertTrue(err("name", name=123))

    def test_unknown_fields_ignored(self):
        fields = dict(GOOD)
        fields["admin"] = "yes"
        self.assertEqual(validate(fields), {})

    def test_never_raises(self):
        for bad in (None, {}, 5, "text", []):
            self.assertEqual(validate(bad), validate(bad))
        self.assertTrue(validate(None)["email"])


class TestClean(unittest.TestCase):
    def test_normalises_known_fields(self):
        c = clean({"email": " A@B.CO ", "password": "abcdefgh12", "age": 30,
                   "name": " Ada ", "extra": 1})
        self.assertEqual(c, {"email": "a@b.co", "password": "abcdefgh12",
                             "age": 30, "name": "Ada"})

    def test_absent_fields_stay_absent(self):
        self.assertEqual(clean({"email": "a@b.co"}),
                         {"email": "a@b.co"})

    def test_never_raises(self):
        for bad in (None, {}, 7, []):
            self.assertIsInstance(clean(bad), dict)


if __name__ == "__main__":
    unittest.main()
