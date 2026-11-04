"""Profile field validation for the accounts service."""


class ValidationError(ValueError):
    pass


def _check_text(value, field, low, high):
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    value = value.strip()
    if len(value) < low or len(value) > high:
        raise ValidationError(f"{field} length must be {low}-{high}")
    return value


def validate_name(value):
    return _check_text(value, "name", 1, 80)


def validate_username(value):
    return _check_text(value, "username", 3, 20)


def validate_bio(value):
    return _check_text(value, "bio", 0, 280)
