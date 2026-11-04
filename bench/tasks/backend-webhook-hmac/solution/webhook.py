"""Webhook signature verification for the payments integration."""
import hashlib
import hmac
import json


def _header(headers, name):
    try:
        value = headers.get(name)
        if value is not None:
            return value
    except AttributeError:
        return None
    lowered = name.lower()
    for key, value in getattr(headers, "items", lambda: [])():
        if key.lower() == lowered:
            return value
    return None


def verify(headers, body, secret, now, tolerance=300):
    sig = _header(headers, "X-Signature")
    ts = _header(headers, "X-Timestamp")
    if not isinstance(sig, str) or not isinstance(ts, str):
        return None
    scheme, _, hex_digest = sig.partition("=")
    if scheme != "sha256" or not hex_digest:
        return None
    try:
        timestamp = int(ts)
    except ValueError:
        return None
    if abs(now - timestamp) > tolerance:
        return None
    if isinstance(body, str):
        raw = body.encode()
    elif isinstance(body, (bytes, bytearray)):
        raw = bytes(body)
    else:
        return None
    key = secret.encode() if isinstance(secret, str) else secret
    expected = hmac.new(key, str(timestamp).encode() + b"." + raw,
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(hex_digest, expected):
        return None
    try:
        event = json.loads(raw)
    except Exception:
        return None
    if not isinstance(event, dict) or "id" not in event:
        return None
    return {"event_id": event["id"], "event": event}
