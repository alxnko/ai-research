"""Bulk user import for the admin console."""
import csv
import io


def _valid_email(value):
    if value.count("@") != 1:
        return False
    local, _, domain = value.partition("@")
    domain = domain.strip(".")
    return bool(local) and "." in domain and " " not in value


def import_users(text):
    if text is None:
        raise ValueError("missing header row")
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("missing header row")
    header = [cell.strip().lower() for cell in rows[0]]
    if "email" not in header or "name" not in header:
        raise ValueError("header must contain 'email' and 'name'")
    email_at = header.index("email")
    name_at = header.index("name")

    accepted, rejected, seen = [], [], set()
    for n, row in enumerate(rows[1:], start=1):
        email = row[email_at].strip() if email_at < len(row) else ""
        name = row[name_at].strip() if name_at < len(row) else ""
        if not email:
            rejected.append({"row": n, "reason": "missing_email"})
            continue
        if not _valid_email(email):
            rejected.append({"row": n, "reason": "bad_email"})
            continue
        if not name:
            rejected.append({"row": n, "reason": "missing_name"})
            continue
        key = email.lower()
        if key in seen:
            rejected.append({"row": n, "reason": "duplicate"})
            continue
        seen.add(key)
        accepted.append({"email": key, "name": name})
    return {"accepted": accepted, "rejected": rejected}
