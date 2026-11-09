"""Failure path retires the in-flight entry before signaling the waiters."""

import threading
import time

_GUARD = threading.Lock()
# (id(store), key) -> slot, present only while a call for that pair is running.
# The slot holds the store, so its id cannot be reused until the call is done.
_INFLIGHT = {}


def run_once(store, key, fn):
    ident = (id(store), key)
    with _GUARD:
        if key in store:
            return store[key]
        slot = _INFLIGHT.get(ident)
        owner = slot is None
        if owner:
            slot = _INFLIGHT[ident] = {
                "event": threading.Event(),
                "store": store,
                "value": None,
                "error": None,
            }
    if owner:
        try:
            value = fn()
        except BaseException as exc:
            with _GUARD:
                _INFLIGHT.pop(ident, None)
            time.sleep(0.3)  # the unsignaled gap, widened so a test can land in it
            slot["error"] = exc
            slot["event"].set()
            raise
        with _GUARD:
            store[key] = value
            slot["value"] = value
            _INFLIGHT.pop(ident, None)
            slot["event"].set()
        return value
    slot["event"].wait()
    if slot["error"] is not None:
        raise slot["error"]
    return slot["value"]
