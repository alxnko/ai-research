"""Stores a placeholder when fn raises, so a retry never runs fn again."""

import threading

_GUARD = threading.Lock()


def run_once(store, key, fn):
    with _GUARD:
        if key in store:
            return store[key]
    try:
        value = fn()
    except BaseException:
        with _GUARD:
            store[key] = None
        raise
    with _GUARD:
        store[key] = value
    return value
