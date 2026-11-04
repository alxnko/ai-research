"""Filename ordering for the asset browser."""
import unicodedata


def _fold(name):
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in decomposed
                       if not unicodedata.combining(c))
    return stripped.casefold()


def sort_names(names):
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"names must be strings, got {type(name).__name__}")
    return sorted(names, key=lambda n: (_fold(n), n))
