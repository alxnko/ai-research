"""Offline mutation queue for the mobile client."""


class SyncQueue:
    def push(self, kind, payload=None):
        raise NotImplementedError("push is not implemented yet")

    def pending(self):
        raise NotImplementedError("pending is not implemented yet")

    def ack(self, mutation_id):
        raise NotImplementedError("ack is not implemented yet")

    def retry(self, mutation_id):
        raise NotImplementedError("retry is not implemented yet")

    def __len__(self):
        raise NotImplementedError("__len__ is not implemented yet")

    def to_dict(self):
        raise NotImplementedError("to_dict is not implemented yet")


def from_dict(data):
    raise NotImplementedError("from_dict is not implemented yet")
