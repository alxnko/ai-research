"""Short-lived signed tokens for the internal API."""
import base64
import hashlib
import hmac
import json

SECRET = b"dev-secret-do-not-use-in-prod"


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(payload_b64):
    return hmac.new(SECRET, payload_b64.encode(), hashlib.sha256).hexdigest()


def issue(uid, roles, now, ttl=3600):
    payload = {"uid": uid, "roles": sorted(roles), "exp": now + ttl}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    body = _b64(raw)
    return f"{body}.{_sign(body)}"


def verify(token, now):
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    body, sig = token.split(".")
    if not body or not sig:
        return None
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        pad = "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(body + pad))
        uid, roles, exp = payload["uid"], payload["roles"], payload["exp"]
    except Exception:
        return None
    if not isinstance(exp, int) or now >= exp:
        return None
    if not isinstance(roles, list):
        return None
    return {"uid": uid, "roles": sorted(roles)}
