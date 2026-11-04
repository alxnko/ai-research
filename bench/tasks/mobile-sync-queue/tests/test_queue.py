import json
import unittest

from queue import SyncQueue, from_dict


class TestSyncQueue(unittest.TestCase):
    def test_starts_empty(self):
        q = SyncQueue()
        self.assertEqual(q.pending(), [])
        self.assertEqual(len(q), 0)

    def test_push_keeps_order_and_ids(self):
        q = SyncQueue()
        a = q.push("create", {"n": 1})
        b = q.push("delete", {"n": 2})
        self.assertNotEqual(a, b)
        self.assertIsInstance(a, str)
        self.assertEqual([m["kind"] for m in q.pending()], ["create", "delete"])
        self.assertEqual(len(q), 2)
        self.assertEqual(q.pending()[0]["payload"], {"n": 1})

    def test_pending_is_a_copy(self):
        q = SyncQueue()
        q.push("create", {"n": 1})
        p = q.pending()
        p.append({"id": "fake", "kind": "x", "payload": None})
        p[0]["kind"] = "tampered"
        p[0]["payload"]["n"] = 99
        self.assertEqual(len(q), 1)
        self.assertEqual(q.pending()[0]["kind"], "create")
        self.assertEqual(q.pending()[0]["payload"], {"n": 1})

    def test_ack(self):
        q = SyncQueue()
        mid = q.push("x")
        self.assertTrue(q.ack(mid))
        self.assertFalse(q.ack(mid))
        self.assertEqual(len(q), 0)

    def test_retry_moves_to_back(self):
        q = SyncQueue()
        q.push("first")
        q.push("second")
        first = q.pending()[0]["id"]
        self.assertTrue(q.retry(first))
        self.assertEqual([m["kind"] for m in q.pending()], ["second", "first"])
        self.assertFalse(q.retry("nope"))
        self.assertEqual(len(q), 2)

    def test_json_round_trip(self):
        q = SyncQueue()
        q.push("a", {"n": 1})
        q.push("b", {"n": 2})
        clone = from_dict(json.loads(json.dumps(q.to_dict())))
        self.assertEqual(clone.pending(), q.pending())
        self.assertEqual(len(clone), 2)

    def test_ids_unique_after_restore(self):
        q = SyncQueue()
        q.push("a")
        clone = from_dict(json.loads(json.dumps(q.to_dict())))
        clone.push("b")
        ids = [m["id"] for m in clone.pending()]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(clone), 2)

    def test_ids_unique_after_churn(self):
        q = SyncQueue()
        ids = [q.push("x") for _ in range(5)]
        q.ack(ids[1])
        q.retry(ids[0])
        more = [q.push("y") for _ in range(3)]
        allids = [m["id"] for m in q.pending()]
        self.assertEqual(len(set(allids)), len(allids))
        self.assertFalse(set(ids) & set(more))

    def test_from_dict_empty(self):
        self.assertEqual(from_dict({}).pending(), [])
        self.assertEqual(from_dict(None).pending(), [])


if __name__ == "__main__":
    unittest.main()
