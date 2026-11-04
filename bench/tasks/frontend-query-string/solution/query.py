"""Query-string helpers for the web client's filter state."""
from urllib.parse import quote_plus, unquote_plus


def parse(qs):
    if not qs:
        return {}
    text = qs[1:] if qs.startswith("?") else qs
    out = {}
    for pair in text.split("&"):
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        key = unquote_plus(key)
        value = unquote_plus(value) if sep else ""
        if key in out:
            if isinstance(out[key], list):
                out[key].append(value)
            else:
                out[key] = [out[key], value]
        else:
            out[key] = value
    return out


def build(params):
    parts = []
    for key in sorted(params):
        value = params[key]
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v is None:
                continue
            parts.append(f"{quote_plus(str(key))}={quote_plus(str(v))}")
    return "&".join(parts)


def merge(existing_qs, updates):
    current = parse(existing_qs)
    pairs = []
    for key, value in current.items():
        if key in updates:
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            pairs.append(f"{quote_plus(key)}={quote_plus(str(v))}")
    for key in sorted(updates):
        value = updates[key]
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v is None:
                continue
            pairs.append(f"{quote_plus(key)}={quote_plus(str(v))}")
    return "&".join(pairs)
