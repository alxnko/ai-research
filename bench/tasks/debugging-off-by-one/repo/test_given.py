import unittest
from chunker import chunks


class TestChunks(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(chunks(b"abcdef", 3), [b"abc", b"def"])

    def test_last_partial(self):
        self.assertEqual(chunks(b"abcdefg", 3), [b"abc", b"def", b"g"])


if __name__ == "__main__":
    unittest.main()
