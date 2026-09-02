import json
import tempfile
import unittest
from pathlib import Path

from compare_framework_generators import (
    _adapter_identity,
    build_comparison_record,
    load_resume_prefix,
)
from framework_opd.rollout import FrameworkGenerationResult


def framework_result(*, fallback: bool = False) -> FrameworkGenerationResult:
    return FrameworkGenerationResult(
        steps=("Identify the relationship", "Apply the required operation"),
        attempts=2,
        used_fallback=fallback,
        validation_errors=("numeric_literal",) if fallback else (),
        prompt_tokens=31,
        generated_tokens=17,
        hit_max_attempts=0,
        last_ended_with_eos=True,
        closed_tag=True,
    )


class CompareFrameworkGeneratorsTest(unittest.TestCase):
    def test_adapter_identity_covers_weights_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")

            identity = _adapter_identity(directory)

            self.assertEqual(identity["path"], str(root.resolve()))
            self.assertEqual(len(identity["adapter_model_sha256"]), 64)
            self.assertEqual(len(identity["adapter_config_sha256"]), 64)

    def test_builds_aligned_student_teacher_record(self):
        run_config = {"temperature": 0.0, "max_new_tokens": 128}
        record = build_comparison_record(
            example_id=7,
            source_record={"question": "How many remain?", "answer": "#### 4"},
            student_result=framework_result(),
            teacher_result=framework_result(fallback=True),
            run_config=run_config,
        )

        self.assertEqual(record["example_id"], 7)
        self.assertEqual(record["question"], "How many remain?")
        self.assertEqual(record["reference_answer"], "#### 4")
        self.assertEqual(record["run_config"], run_config)
        self.assertEqual(
            record["student"]["framework"],
            ["Identify the relationship", "Apply the required operation"],
        )
        self.assertFalse(record["student"]["used_fallback"])
        self.assertTrue(record["teacher"]["used_fallback"])
        self.assertEqual(record["teacher"]["validation_errors"], ["numeric_literal"])

    def test_resume_accepts_only_an_exact_dataset_and_config_prefix(self):
        records = [
            {"question": "Q0", "answer": "A0"},
            {"question": "Q1", "answer": "A1"},
        ]
        run_config = {"temperature": 0.0}
        first = build_comparison_record(
            example_id=0,
            source_record=records[0],
            student_result=framework_result(),
            teacher_result=framework_result(),
            run_config=run_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.jsonl"
            output.write_text(json.dumps(first) + "\n", encoding="utf-8")

            loaded = load_resume_prefix(output, records, run_config)
            self.assertEqual(loaded, [first])

            first["question"] = "different question"
            output.write_text(json.dumps(first) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset prefix"):
                load_resume_prefix(output, records, run_config)

    def test_resume_rejects_changed_generation_configuration(self):
        records = [{"question": "Q0", "answer": "A0"}]
        first = build_comparison_record(
            example_id=0,
            source_record=records[0],
            student_result=framework_result(),
            teacher_result=framework_result(),
            run_config={"temperature": 0.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.jsonl"
            output.write_text(json.dumps(first) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run configuration"):
                load_resume_prefix(output, records, {"temperature": 0.7})


if __name__ == "__main__":
    unittest.main()
