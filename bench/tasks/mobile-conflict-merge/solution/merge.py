"""Offline sync conflict resolution for the mobile client."""
import copy


def merge(local, remote):
    if local is None and remote is None:
        return None
    if remote is None:
        return copy.deepcopy(local)
    if local is None:
        return copy.deepcopy(remote)
    if local["updated_at"] > remote["updated_at"]:
        return copy.deepcopy(local)
    winner = copy.deepcopy(remote)
    if local["updated_at"] == remote["updated_at"]:
        winner["conflict"] = True
    return winner


def _by_id(records):
    out = {}
    for rec in records or []:
        current = out.get(rec["id"])
        if current is None or rec["updated_at"] > current["updated_at"]:
            out[rec["id"]] = rec
    return out


def merge_all(local_list, remote_list):
    local = _by_id(local_list)
    remote = _by_id(remote_list)
    ids = sorted(set(local) | set(remote))
    return [merge(local.get(i), remote.get(i)) for i in ids]
