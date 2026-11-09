"""Per-(store, key) lock, but a waiter runs fn itself when the owner failed."""

import threading

_GUARD = threading.Lock()
_LOCKS = {}


def run_once(store, key, fn):
    with _GUARD:
        lock = _LOCKS.setdefault((id(store), key), threading.Lock())
    with lock:
        if key in store:
            return store[key]
        value = fn()
        store[key] = value
        return value
