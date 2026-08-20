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
        self.assertEqual(vanilla["beta"], 1.0)
        self.assertEqual(guided["beta"], 1.0)
        self.assertEqual(vanilla["max_steps"] % vanilla["gradient_accumulation_steps"], 0)
        self.assertGreaterEqual(vanilla["solution_max_new_tokens"], 512)
        self.assertIn("-v2", vanilla["output_dir"])
        self.assertIn("-v2", guided["output_dir"])
        self.assertIn("-v2", guided["framework_teacher_adapter"])

    def test_evaluation_uses_only_v2_artifacts_and_a_larger_answer_budget(self):
        root = Path(__file__).parents[1]
        evaluation = json.loads((root / "configs/evaluation.json").read_text())
        self.assertIn("-v2", evaluation["output_dir"])
        self.assertIn("-v2", evaluation["framework_teacher_adapter"])
        self.assertTrue(all("-v2" in path for path in evaluation["adapters"].values()))
        self.assertGreaterEqual(evaluation["max_new_tokens"], 1024)
