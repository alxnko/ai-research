"""Chunked upload helper for the asset service."""


def chunks(data, size):
    if not isinstance(data, (bytes, bytearray, str)):
        raise TypeError("data must be bytes or str")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be positive")
    return [data[i:i + size] for i in range(0, len(data), size)]
