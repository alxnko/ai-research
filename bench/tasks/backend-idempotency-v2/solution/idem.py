"""Idempotency support for retried write requests."""

import threading

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
        # Whatever happens, retire the entry and wake the waiters under the
        # guard, so none of them can block on a call that no longer runs.
        try:
            value = fn()
            with _GUARD:
                store[key] = value
                slot["value"] = value
        except BaseException as exc:
            slot["error"] = exc
            raise
        finally:
            with _GUARD:
                _INFLIGHT.pop(ident, None)
                slot["event"].set()
        return value
    slot["event"].wait()
    if slot["error"] is not None:
        raise slot["error"]
    return slot["value"]
