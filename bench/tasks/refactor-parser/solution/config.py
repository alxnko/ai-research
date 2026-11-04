"""Config file parsing for the CLI."""


class ConfigError(ValueError):
    pass


def parse_kv(text, sep):
    out = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if sep not in raw:
            raise ConfigError(f"bad line: {line!r}")
        key, _, value = raw.partition(sep)
        out[key.strip()] = value.strip()
    return out


def parse_equals(text):
    return parse_kv(text, "=")


def parse_colon(text):
    return parse_kv(text, ":")
