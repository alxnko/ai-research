import copy
import unittest

from merge import merge, merge_all


def rec(rid, ts, **fields):
    return {"id": rid, "updated_at": ts, "fields": fields}


class TestMerge(unittest.TestCase):
    def test_remote_newer_wins(self):
        out = merge(rec("a", 1, v="old"), rec("a", 2, v="new"))
        self.assertEqual(out["fields"], {"v": "new"})
        self.assertNotIn("conflict", out)

    def test_local_newer_wins(self):
        out = merge(rec("a", 3, v="local"), rec("a", 2, v="remote"))
        self.assertEqual(out["fields"], {"v": "local"})
        self.assertNotIn("conflict", out)

    def test_tie_goes_to_remote_with_flag(self):
        out = merge(rec("a", 5, v="local"), rec("a", 5, v="remote"))
        self.assertEqual(out["fields"], {"v": "remote"})
        self.assertIs(out["conflict"], True)

    def test_one_sided(self):
        self.assertEqual(merge(rec("a", 1, v="l"), None)["fields"], {"v": "l"})
        self.assertEqual(merge(None, rec("b", 1, v="r"))["fields"], {"v": "r"})
        self.assertIsNone(merge(None, None))

    def test_inputs_not_mutated(self):
        local = rec("a", 1, tags=["x"])
        remote = rec("a", 2, tags=["y"])
        snap = copy.deepcopy((local, remote))
        out = merge(local, remote)
        self.assertEqual((local, remote), snap)
        out["fields"]["tags"].append("z")
        self.assertEqual(remote["fields"]["tags"], ["y"])

    def test_result_is_a_copy(self):
        local = rec("a", 1, n=1)
        out = merge(local, None)
        out["fields"]["n"] = 99
        self.assertEqual(local["fields"]["n"], 1)


class TestMergeAll(unittest.TestCase):
    def test_sorted_by_id(self):
        out = merge_all([rec("b", 1)], [rec("a", 1), rec("c", 1)])
        self.assertEqual([r["id"] for r in out], ["a", "b", "c"])

    def test_pairwise_merge(self):
        out = merge_all([rec("a", 1, v="l")], [rec("a", 2, v="r")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["fields"], {"v": "r"})

    def test_duplicates_keep_newest(self):
        out = merge_all([rec("a", 1, v="old"), rec("a", 4, v="new")], [])
        self.assertEqual(out[0]["fields"], {"v": "new"})

    def test_conflict_flag_survives_merge_all(self):
        out = merge_all([rec("a", 7, v="l")], [rec("a", 7, v="r")])
        self.assertIs(out[0]["conflict"], True)

    def test_empty_inputs(self):
        self.assertEqual(merge_all([], []), [])
        self.assertEqual(merge_all(None, None), [])


if __name__ == "__main__":
    unittest.main()
