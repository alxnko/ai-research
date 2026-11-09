"""Check-then-act with no synchronisation: concurrent callers all run fn."""


def run_once(store, key, fn):
    if key in store:
        return store[key]
    value = fn()
    store[key] = value
    return value
