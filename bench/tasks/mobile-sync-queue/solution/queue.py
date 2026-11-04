"""Offline mutation queue for the mobile client."""

import copy


class SyncQueue:
    def __init__(self, items=None, next_id=1):
        self._items = [dict(m) for m in (items or [])]
        self._next = int(next_id)

    def push(self, kind, payload=None):
        mid = f"m{self._next}"
        self._next += 1
        self._items.append({"id": mid, "kind": kind, "payload": copy.deepcopy(payload)})
        return mid

    def pending(self):
        return copy.deepcopy(self._items)

    def ack(self, mutation_id):
        for i, m in enumerate(self._items):
            if m["id"] == mutation_id:
                del self._items[i]
                return True
        return False

    def retry(self, mutation_id):
        for i, m in enumerate(self._items):
            if m["id"] == mutation_id:
                self._items.append(self._items.pop(i))
                return True
        return False

    def __len__(self):
        return len(self._items)

    def to_dict(self):
        return {"items": copy.deepcopy(self._items), "next_id": self._next}


def from_dict(data):
    d = data or {}
    return SyncQueue(d.get("items", []), d.get("next_id", 1))
