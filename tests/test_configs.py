import json
import unittest
from pathlib import Path


class ConfigTest(unittest.TestCase):
    def test_controlled_configs_match(self):
        root = Path(__file__).parents[1]
        vanilla = json.loads((root / "configs/vanilla_opd.json").read_text())
        guided = json.loads((root / "configs/guided_opd.json").read_text())
        allowed = {"mode", "output_dir", "framework_teacher_adapter"}
        keys = (set(vanilla) | set(guided)) - allowed
        self.assertEqual({key: vanilla.get(key) for key in keys}, {key: guided.get(key) for key in keys})
