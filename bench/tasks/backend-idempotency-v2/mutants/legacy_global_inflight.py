"""Idempotency support for retried write requests."""

import threading

_GUARD = threading.Lock()
_INFLIGHT = {}
_DONE = "__idem_done__"


def run_once(store, key, fn):
    with _GUARD:
        if key in store:
            return store[key]
        slot = _INFLIGHT.get(key)
        owner = slot is None
        if owner:
            slot = _INFLIGHT[key] = {"event": threading.Event(),
                                     "value": None, "error": None}
    if owner:
        try:
            value = fn()
        except BaseException as exc:
            with _GUARD:
                _INFLIGHT.pop(key, None)
            slot["error"] = exc
            slot["event"].set()
            raise
        with _GUARD:
            store[key] = value
            _INFLIGHT.pop(key, None)
        slot["value"] = value
        slot["event"].set()
        return value
    slot["event"].wait()
    if slot["error"] is not None:
        raise slot["error"]
    return store[key]
