"""The extra expert caches setup picks beside the GPUs the model runs on."""
import unittest

import setup


CARDS = [
    {"index": 0, "name": "main", "vram_gb": 12, "arch": "86", "driver": "580.0"},
    {"index": 1, "name": "extra", "vram_gb": 6, "arch": "75", "driver": "580.0"},
    {"index": 2, "name": "old", "vram_gb": 24, "arch": "75", "driver": "580.0"},
    {"index": 3, "name": "tiny", "vram_gb": 3, "arch": "86", "driver": "580.0"},
    {"index": 4, "name": "pascal", "vram_gb": 11, "arch": "61", "driver": "580.0"},
]


class ExtraGpus(unittest.TestCase):
    def test_the_rest_by_vram(self):
        self.assertEqual([g["index"] for g in setup.extra_gpus(CARDS, [0])], [2, 1])

    def test_the_split_cards_are_not_extras(self):
        self.assertEqual([g["index"] for g in setup.extra_gpus(CARDS, [0, 2])], [1])


if __name__ == "__main__":
    unittest.main()
