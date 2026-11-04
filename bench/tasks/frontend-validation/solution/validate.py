"""Signup form validation shared by the web and mobile clients."""

KNOWN = ("email", "password", "age", "name")


def _text(value):
    return value if isinstance(value, str) else ""


def _valid_email(value):
    s = value.strip()
    if s.count("@") != 1:
        return False
    local, _, domain = s.partition("@")
    return bool(local) and "." in domain.strip(".") and " " not in s


def validate(fields):
    """Return a dict of field name -> error message. Empty when all is valid."""
    f = fields if isinstance(fields, dict) else {}
    errors = {}

    email = _text(f.get("email"))
    if not email.strip():
        errors["email"] = "email is required"
    elif not _valid_email(email):
        errors["email"] = "email is invalid"

    password = _text(f.get("password"))
    if not password:
        errors["password"] = "password is required"
    elif len(password) < 10:
        errors["password"] = "password must be at least 10 characters"
    elif not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        errors["password"] = "password must contain a letter and a digit"

    age = f.get("age")
    if age is None:
        errors["age"] = "age is required"
    elif isinstance(age, bool) or not isinstance(age, int):
        errors["age"] = "age must be a whole number"
    elif age < 13 or age > 120:
        errors["age"] = "age must be between 13 and 120"

    name = f.get("name")
    if name is not None:
        if not isinstance(name, str):
            errors["name"] = "name must be text"
        elif len(name.strip()) > 80:
            errors["name"] = "name is too long"

    return errors


def clean(fields):
    """Return only the known fields, normalised."""
    f = fields if isinstance(fields, dict) else {}
    out = {}
    if "email" in f:
        out["email"] = _text(f["email"]).strip().lower()
    if "password" in f:
        out["password"] = f["password"]
    if "age" in f:
        out["age"] = f["age"]
    if "name" in f:
        name = f["name"]
        out["name"] = name.strip() if isinstance(name, str) else name
    return out
