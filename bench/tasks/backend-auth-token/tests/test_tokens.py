import base64
import json
import unittest

import tokens
from tokens import issue, verify

NOW = 1_700_000_000


class TestTokens(unittest.TestCase):
    def test_round_trip(self):
        t = issue("u1", ["reader", "writer"], NOW)
        self.assertEqual(verify(t, NOW), {"uid": "u1", "roles": ["reader", "writer"]})

    def test_shape(self):
        t = issue("u1", [], NOW)
        self.assertEqual(t.count("."), 1)
        body = t.split(".")[0]
        self.assertNotIn("=", body)
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        self.assertEqual(payload["exp"], NOW + 3600)
        self.assertEqual(sorted(payload), ["exp", "roles", "uid"])

    def test_custom_ttl_and_expiry(self):
        t = issue("u", [], NOW, ttl=10)
        self.assertIsNotNone(verify(t, NOW + 9))
        self.assertIsNone(verify(t, NOW + 10))
        self.assertIsNone(verify(t, NOW + 11))

    def test_roles_sorted(self):
        t = issue("u", ["z", "a", "m"], NOW)
        self.assertEqual(verify(t, NOW)["roles"], ["a", "m", "z"])

    def test_tampered_payload_rejected(self):
        t = issue("u1", ["reader"], NOW)
        body, sig = t.split(".")
        forged = base64.urlsafe_b64encode(
            json.dumps({"uid": "admin", "roles": ["admin"], "exp": NOW + 3600},
                       sort_keys=True, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        self.assertIsNone(verify(f"{forged}.{sig}", NOW))

    def test_tampered_signature_rejected(self):
        t = issue("u1", [], NOW)
        body, sig = t.split(".")
        bad = ("0" if sig[0] != "0" else "1") + sig[1:]
        self.assertIsNone(verify(f"{body}.{bad}", NOW))

    def test_malformed_rejected(self):
        for bad in ["", ".", "abc", "a.b.c", "!!.??", "abc.def", None, 5]:
            self.assertIsNone(verify(bad, NOW))

    def test_constant_time_compare_used(self):
        import inspect
        src = inspect.getsource(tokens)
        self.assertTrue("compare_digest" in src or "secrets.compare_digest" in src)


if __name__ == "__main__":
    unittest.main()
