"""Short-lived signed tokens for the internal API."""
import base64
import hashlib
import hmac
import json

SECRET = b"dev-secret-do-not-use-in-prod"


def issue(uid, roles, now, ttl=3600):
    raise NotImplementedError("issue is not implemented yet")


def verify(token, now):
    raise NotImplementedError("verify is not implemented yet")
