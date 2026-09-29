"""GPU selection shared by setup's --gpu option and multi-GPU configs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup


CARDS = [
    {"index": 2, "name": "old", "vram_gb": 24, "arch": "75", "driver": "580.0", "uuid": "GPU-old"},
    {"index": 0, "name": "main", "vram_gb": 12, "arch": "86", "driver": "580.0", "uuid": "GPU-main"},
    {"index": 1, "name": "extra", "vram_gb": 6, "arch": "75", "driver": "580.0", "uuid": "GPU-extra"},
]


class GpuSelection(unittest.TestCase):
    def setUp(self):
        self.cards = patch.object(setup, "gpus", return_value=CARDS)
        self.cards.start()
        self.addCleanup(self.cards.stop)
        self.pick = patch.object(setup, "GPU_PICK", None)
        self.pick.start()
        self.addCleanup(self.pick.stop)

    def test_default_and_explicit_index(self):
        self.assertEqual(setup.gpu_info()["uuid"], "GPU-main")
        self.assertEqual(setup.gpu_info()["count"], 3)
        self.assertEqual(setup.gpu_info(2)["uuid"], "GPU-old")

    def test_installed_uuid_order_and_extra_arch(self):
        cfg = {"gpu": 0, "env": {"CUDA_VISIBLE_DEVICES": "GPU-main,GPU-extra"},
               "args": ["--extra-gpus", "1"]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps(cfg))
            self.assertEqual(setup.configured_gpu(path)["uuid"], "GPU-main")
            self.assertEqual(setup.configured_extra_archs(path, CARDS[1]), ["75"])
            self.assertEqual(setup.configured_gpu(path, 2)["uuid"], "GPU-old")
            cfg["env"] = {"CUDA_VISIBLE_DEVICES": "2"}
            path.write_text(json.dumps(cfg))
            self.assertEqual(setup.configured_gpu(path)["uuid"], "GPU-old")


if __name__ == "__main__":
    unittest.main()
