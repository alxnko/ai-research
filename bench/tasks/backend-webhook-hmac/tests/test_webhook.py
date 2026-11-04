import hashlib
import hmac
import json
import unittest

from webhook import verify

SECRET = "whsec_test"
NOW = 1_700_000_000
BODY = json.dumps({"id": "evt_1", "type": "charge.succeeded"})


def sign(body, ts=NOW, secret=SECRET):
    return hmac.new(secret.encode(), f"{ts}.{body}".encode(),
                    hashlib.sha256).hexdigest()


def headers(body=BODY, ts=NOW, sig=None, scheme="sha256"):
    return {"X-Signature": f"{scheme}={sig or sign(body, ts)}",
            "X-Timestamp": str(ts)}


class TestVerify(unittest.TestCase):
    def test_accepts_valid(self):
        out = verify(headers(), BODY, SECRET, NOW)
        self.assertEqual(out["event_id"], "evt_1")
        self.assertEqual(out["event"]["type"], "charge.succeeded")

    def test_case_insensitive_headers(self):
        h = {k.lower(): v for k, v in headers().items()}
        self.assertIsNotNone(verify(h, BODY, SECRET, NOW))

    def test_bytes_body(self):
        self.assertIsNotNone(verify(headers(), BODY.encode(), SECRET, NOW))

    def test_tampered_body(self):
        self.assertIsNone(verify(headers(), BODY.replace("evt_1", "evt_2"),
                                 SECRET, NOW))

    def test_wrong_secret(self):
        self.assertIsNone(verify(headers(), BODY, "other", NOW))

    def test_missing_headers(self):
        self.assertIsNone(verify({}, BODY, SECRET, NOW))
        self.assertIsNone(verify({"X-Signature": headers()["X-Signature"]},
                                 BODY, SECRET, NOW))
        self.assertIsNone(verify({"X-Timestamp": str(NOW)}, BODY, SECRET, NOW))

    def test_bad_scheme(self):
        self.assertIsNone(verify(headers(scheme="sha1"), BODY, SECRET, NOW))

    def test_bad_hex(self):
        h = headers()
        h["X-Signature"] = "sha256=zzzz"
        self.assertIsNone(verify(h, BODY, SECRET, NOW))

    def test_timestamp_window(self):
        self.assertIsNotNone(verify(headers(ts=NOW - 300), BODY, SECRET, NOW))
        self.assertIsNotNone(verify(headers(ts=NOW + 300), BODY, SECRET, NOW))
        self.assertIsNone(verify(headers(ts=NOW - 301), BODY, SECRET, NOW))
        self.assertIsNone(verify(headers(ts=NOW + 301), BODY, SECRET, NOW))

    def test_bad_timestamp(self):
        h = headers()
        h["X-Timestamp"] = "soon"
        self.assertIsNone(verify(h, BODY, SECRET, NOW))

    def test_signature_bound_to_timestamp(self):
        h = headers(ts=NOW)
        h["X-Timestamp"] = str(NOW - 10)
        self.assertIsNone(verify(h, BODY, SECRET, NOW))

    def test_invalid_json_and_missing_id(self):
        for bad in ("not json", "[]", '{"type": "x"}'):
            h = headers(body=bad)
            self.assertIsNone(verify(h, bad, SECRET, NOW))

    def test_constant_time(self):
        import inspect
        import webhook
        self.assertIn("compare_digest", inspect.getsource(webhook))


if __name__ == "__main__":
    unittest.main()
