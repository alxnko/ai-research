"""Per-session message cache for the chat service."""


class SessionCache:
    def __init__(self, history=None, limit=100):
        self.history = list(history) if history else []
        self.limit = limit

    def add(self, message):
        self.history.append(message)
        if len(self.history) > self.limit:
            del self.history[: len(self.history) - self.limit]
        return len(self.history)

    def snapshot(self):
        return list(self.history)

    def clear(self):
        self.history = []
