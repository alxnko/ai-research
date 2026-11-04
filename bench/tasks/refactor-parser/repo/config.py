"""Config file parsing for the CLI."""


class ConfigError(ValueError):
    pass


def parse_equals(text):
    out = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            raise ConfigError(f"bad line: {line!r}")
        key, _, value = raw.partition("=")
        out[key.strip()] = value.strip()
    return out


def parse_colon(text):
    out = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            raise ConfigError(f"bad line: {line!r}")
        key, _, value = raw.partition(":")
        out[key.strip()] = value.strip()
    return out
