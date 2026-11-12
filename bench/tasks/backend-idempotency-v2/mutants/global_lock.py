"""One process-wide lock around fn: correct per key, but every call waits."""

import threading

_LOCK = threading.RLock()


def run_once(store, key, fn):
    with _LOCK:
        if key in store:
            return store[key]
        value = fn()
        store[key] = value
        return value
