"""Contact formatting for the CRM export."""


def normalize_phone(value):
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"+1{digits}"


def normalize_home(value):
    return normalize_phone(value)


def normalize_work(value):
    return normalize_phone(value)


def format_contact(contact):
    home = normalize_home(contact.get("home", ""))
    work = normalize_work(contact.get("work", ""))
    return {
        "name": contact.get("name", "").strip(),
        "home": home,
        "work": work,
    }
