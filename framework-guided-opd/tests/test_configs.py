import json
import unittest
from pathlib import Path


class ConfigTest(unittest.TestCase):
    def test_runtime_entrypoints_are_relocatable(self):
        root = Path(__file__).parents[1]
        old_root = "/root/blockdata/framework-guided-opd"
        for name in (
            "run_framework_data.sh",
            "run_teacher_training.sh",
            "run_smoke.sh",
            "run_comparison_training.sh",
            "run_evaluation.sh",
        ):
            script = (root / name).read_text(encoding="utf-8")
            self.assertIn('BASH_SOURCE[0]', script, name)
            self.assertIn('PYTHONPATH="$SCRIPT_DIR/src"', script, name)
            self.assertNotIn(old_root, script, name)

    def test_project_artifact_paths_are_relative(self):
        root = Path(__file__).parents[1]
        for name in (
            "evaluation.json",
            "guided_opd.json",
            "smoke.json",
            "vanilla_opd.json",
            "vanilla_smoke.json",
        ):
            config = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
            for key in ("framework_teacher_adapter", "output_dir"):
                if config.get(key):
                    self.assertFalse(Path(config[key]).is_absolute(), f"{name}:{key}")
            for arm, path in config.get("adapters", {}).items():
                self.assertFalse(Path(path).is_absolute(), f"{name}:adapters.{arm}")

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

    def test_evaluation_uses_v2_model_artifacts_and_bounded_answer_budget(self):
        root = Path(__file__).parents[1]
        evaluation = json.loads((root / "configs/evaluation.json").read_text())
        self.assertEqual(evaluation["output_dir"], "outputs/comparison-eval-v3-answer-stop")
        self.assertIn("-v2", evaluation["framework_teacher_adapter"])
        self.assertTrue(all("-v2" in path for path in evaluation["adapters"].values()))
        self.assertEqual(evaluation["max_new_tokens"], 2048)
        self.assertLessEqual(evaluation["max_new_tokens"], 2048)
