"""Deep-link routing for the mobile app."""
from urllib.parse import unquote


def _segments(path):
    path = path.split("?", 1)[0].split("#", 1)[0]
    return [s for s in path.strip("/").split("/") if s != ""]


class Router:
    def __init__(self, routes):
        self.routes = []
        seen = set()
        for pattern, name in routes:
            if not pattern.startswith("/"):
                raise ValueError(f"pattern must start with '/': {pattern!r}")
            if pattern in seen:
                raise ValueError(f"duplicate pattern: {pattern!r}")
            seen.add(pattern)
            self.routes.append((_segments(pattern), name))

    def match(self, path):
        parts = _segments(path)
        for pattern, name in self.routes:
            if len(pattern) != len(parts):
                continue
            params = {}
            for want, got in zip(pattern, parts):
                if want.startswith(":"):
                    params[want[1:]] = unquote(got)
                elif want == "*":
                    params["*"] = unquote(got)
                elif want != got:
                    break
            else:
                return {"name": name, "params": params}
        return None
