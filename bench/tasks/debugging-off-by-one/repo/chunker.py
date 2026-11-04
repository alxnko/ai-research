"""Chunked upload helper for the asset service."""


def chunks(data, size):
    if size <= 0:
        raise ValueError("size must be positive")
    out = []
    i = 0
    while i + size < len(data):
        out.append(data[i:i + size])
        i += size
    return out
