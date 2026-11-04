import unittest

from chunker import chunks


class TestChunkEdges(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunks(b"", 4), [])

    def test_smaller_than_chunk(self):
        self.assertEqual(chunks(b"ab", 10), [b"ab"])

    def test_single_byte_chunks(self):
        self.assertEqual(chunks(b"abc", 1), [b"a", b"b", b"c"])

    def test_round_trip(self):
        data = bytes(range(256)) * 3 + b"tail"
        self.assertEqual(b"".join(chunks(data, 7)), data)

    def test_str_supported(self):
        self.assertEqual(chunks("abcdefg", 3), ["abc", "def", "g"])

    def test_bytearray(self):
        self.assertEqual([bytes(c) for c in chunks(bytearray(b"abcd"), 2)],
                         [b"ab", b"cd"])

    def test_invalid_size(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                chunks(b"abc", bad)

    def test_exact_multiple_keeps_all(self):
        self.assertEqual(chunks(b"12345678", 4), [b"1234", b"5678"])
        self.assertEqual(chunks(b"1234", 2), [b"12", b"34"])


if __name__ == "__main__":
    unittest.main()
