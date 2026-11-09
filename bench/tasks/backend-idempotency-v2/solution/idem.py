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
        try:
            value = fn()
        except BaseException as exc:
            # publish the failure and retire the entry in one step, so a new
            # caller either joins this call or starts after it has finished
            with _GUARD:
                slot["error"] = exc
                _INFLIGHT.pop(ident, None)
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
