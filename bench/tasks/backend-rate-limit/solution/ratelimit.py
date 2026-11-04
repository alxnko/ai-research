"""Per-key sliding-window rate limiting for the public API."""


class RateLimiter:
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self._events = {}

    def _live(self, key, now):
        cutoff = now - self.window
        kept = [t for t in self._events.get(key, []) if t > cutoff]
        if kept:
            self._events[key] = kept
        else:
            self._events.pop(key, None)
        return kept

    def allow(self, key, now):
        live = self._live(key, now)
        if len(live) >= self.limit:
            return False
        self._events.setdefault(key, []).append(now)
        return True

    def remaining(self, key, now):
        return max(0, self.limit - len(self._live(key, now)))

    def reset_at(self, key, now):
        live = self._live(key, now)
        return live[0] + self.window if live else None
