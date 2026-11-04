"""Per-session message cache for the chat service."""


class SessionCache:
    def __init__(self, history=[], limit=100):
        self.history = history
        self.limit = limit

    def add(self, message):
        self.history.append(message)
        if len(self.history) > self.limit:
            del self.history[: len(self.history) - self.limit]
        return len(self.history)

    def snapshot(self):
        return self.history

    def clear(self):
        self.history = []
