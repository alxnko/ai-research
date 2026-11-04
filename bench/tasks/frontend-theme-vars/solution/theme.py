"""CSS custom-property resolution for the design-token build."""


def _split_args(body):
    """Split 'a, b' on top-level commas only."""
    parts, depth, current = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return parts


def _substitute(value, tokens, stack):
    out, i = "", 0
    while True:
        start = value.find("var(", i)
        if start == -1:
            out += value[i:]
            break
        out += value[i:start]
        depth, j = 1, start + 4
        while j < len(value) and depth:
            if value[j] == "(":
                depth += 1
            elif value[j] == ")":
                depth -= 1
            j += 1
        if depth:
            raise ValueError("unbalanced var() in theme value")
        body = value[start + 4:j - 1]
        args = _split_args(body)
        ref = args[0].strip()
        fallback = ",".join(args[1:]).strip() if len(args) > 1 else None
        if ref in stack:
            raise ValueError(f"cycle detected at {ref}")
        if ref in tokens:
            out += _resolve(tokens, ref, stack)
        elif fallback is not None:
            out += _substitute(fallback, tokens, stack)
        else:
            raise KeyError(ref)
        i = j
    return out


def _resolve(tokens, name, stack):
    if name in stack:
        raise ValueError(f"cycle detected at {name}")
    value = _substitute(tokens[name], tokens, stack | {name})
    return value.strip()


def resolve(tokens, name):
    return _resolve(tokens, name, frozenset())
