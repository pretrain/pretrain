import unittest

from src.utils.download import get_format


class TestDownload(unittest.TestCase):
    def test_get_format(self):
        self.assertEqual(get_format("gz"), "gztar")
        self.assertEqual(get_format("bz2"), "bztar")
