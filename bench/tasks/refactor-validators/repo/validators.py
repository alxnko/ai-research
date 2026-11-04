"""Profile field validation for the accounts service."""


class ValidationError(ValueError):
    pass


def validate_name(value):
    if not isinstance(value, str):
        raise ValidationError("name must be text")
    value = value.strip()
    if len(value) < 1 or len(value) > 80:
        raise ValidationError("name length must be 1-80")
    return value


def validate_username(value):
    if not isinstance(value, str):
        raise ValidationError("username must be text")
    value = value.strip()
    if len(value) < 3 or len(value) > 20:
        raise ValidationError("username length must be 3-20")
    return value


def validate_bio(value):
    if not isinstance(value, str):
        raise ValidationError("bio must be text")
    value = value.strip()
    if len(value) < 0 or len(value) > 280:
        raise ValidationError("bio length must be 0-280")
    return value
