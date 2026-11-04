"""Signup form validation shared by the web and mobile clients."""


def validate(fields):
    """Return a dict of field name -> error message. Empty when all is valid."""
    raise NotImplementedError("validate is not implemented yet")


def clean(fields):
    """Return only the known fields, normalised."""
    raise NotImplementedError("clean is not implemented yet")
