import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from prepare_framework_data import _publish_generation_result
from framework_opd.evaluation import artifact_fingerprint
from train_teacher import (
    _write_run_metadata,
    adapter_artifact_identity,
    lightweight_model_identity,
    load_framework_training_records,
    main as train_teacher_main,
    prepare_empty_output_directory,
    validate_generation_audit,
    verify_teacher_artifact,
)


class FrameworkDataPublicationTest(unittest.TestCase):
    def test_incomplete_generation_does_not_publish_formal_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "labels.jsonl"
            audit_output = root / "labels.audit.json"
            records = [{"question": "q", "answer": "#### 5", "framework": ["Plan", "Check"]}]

            audit = _publish_generation_result(
                output=output,
                audit_output=audit_output,
                accepted_records=records,
                audit={"requested_valid": 2, "valid": 1},
                complete=False,
            )

            partial = Path(str(output) + ".partial")
            self.assertFalse(output.exists())
            self.assertTrue(partial.exists())
            self.assertEqual(audit["status"], "incomplete")
            self.assertIsNone(audit["output_sha256"])
            self.assertEqual(audit["partial_output"], str(partial))
            self.assertEqual(audit["data_sha256"], hashlib.sha256(partial.read_bytes()).hexdigest())
            self.assertEqual(json.loads(audit_output.read_text(encoding="utf-8")), audit)

    def test_complete_generation_atomically_promotes_partial_to_formal_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "labels.jsonl"
            audit_output = root / "labels.audit.json"
            audit = _publish_generation_result(
                output=output,
                audit_output=audit_output,
                accepted_records=[{"framework": ["Plan", "Check"]}],
                audit={"requested_valid": 1, "valid": 1},
                complete=True,
            )

            self.assertTrue(output.exists())
            self.assertFalse(Path(str(output) + ".partial").exists())
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["requested_valid"], audit["valid"])
            self.assertEqual(audit["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())


class TeacherPreflightTest(unittest.TestCase):
    @staticmethod
    def _write_valid_records(path: Path, count: int) -> None:
        records = [
            {
                "question": f"question {index}",
                "answer": "reasoning\n#### 5",
                "framework": ["Identify the quantities", "Combine them and verify the result"],
            }
            for index in range(count)
        ]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def test_expected_record_count_is_checked_during_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "labels.jsonl"
            self._write_valid_records(data, 1)
            with self.assertRaisesRegex(ValueError, "expected exactly 2"):
                load_framework_training_records(data, expected_records=2)

    @staticmethod
    def _write_generation_audit(data: Path, audit_path: Path, **overrides) -> dict:
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        audit = {
            "status": "complete",
            "requested_valid": 1,
            "valid": 1,
            "output": str(data.resolve()),
            "output_sha256": digest,
            "data_sha256": digest,
            "partial_output": None,
        }
        audit.update(overrides)
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return audit

    def test_generation_audit_binds_complete_formal_data_and_exact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "labels.jsonl"
            audit_path = root / "generation-audit.json"
            self._write_valid_records(data, 1)
            expected = self._write_generation_audit(data, audit_path)

            self.assertEqual(validate_generation_audit(data, audit_path, 1), expected)

    def test_generation_audit_rejects_status_hash_count_and_output_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "labels.jsonl"
            audit_path = root / "generation-audit.json"
            self._write_valid_records(data, 1)
            cases = (
                ({"status": "incomplete"}, "status"),
                ({"output_sha256": "wrong"}, "output_sha256"),
                ({"requested_valid": 2}, "counts"),
                ({"output": str(root / "other.jsonl")}, "output path"),
            )
            for overrides, message in cases:
                with self.subTest(overrides=overrides):
                    self._write_generation_audit(data, audit_path, **overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_generation_audit(data, audit_path, 1)

    def test_partial_data_is_rejected_even_with_a_complete_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "labels.jsonl.partial"
            audit_path = root / "generation-audit.json"
            self._write_valid_records(data, 1)
            self._write_generation_audit(data, audit_path)
            with self.assertRaisesRegex(ValueError, "partial"):
                validate_generation_audit(data, audit_path, 1)

    def test_teacher_output_must_be_absent_or_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher"
            self.assertEqual(prepare_empty_output_directory(output), output)
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "absent or empty"):
                prepare_empty_output_directory(output)

    def test_main_rejects_nonempty_output_before_model_identity_or_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "labels.jsonl"
            audit_path = root / "generation-audit.json"
            output = root / "teacher"
            output.mkdir()
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            self._write_valid_records(data, 1)
            self._write_generation_audit(data, audit_path)
            argv = [
                "train_teacher.py",
                "--model",
                str(root / "missing-model"),
                "--data",
                str(data),
                "--generation-audit",
                str(audit_path),
                "--output",
                str(output),
                "--expected-records",
                "1",
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                train_teacher_main()

    def test_non_positive_steps_fail_before_reading_data_or_loading_model(self):
        argv = [
            "train_teacher.py",
            "--model",
            "unused",
            "--data",
            "missing.jsonl",
            "--generation-audit",
            "missing.audit.json",
            "--output",
            "unused-output",
            "--steps",
            "0",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as context:
            train_teacher_main()
        self.assertEqual(context.exception.code, 2)

    def test_help_exposes_required_generation_audit(self):
        with (
            patch.object(sys, "argv", ["train_teacher.py", "--help"]),
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as context,
        ):
            train_teacher_main()
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--generation-audit", output.getvalue())

    def test_completion_marker_commits_run_config_hash_and_role(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
            weights = output / "adapter_model.safetensors"
            weights.write_bytes(b"adapter-v1")
            artifact = adapter_artifact_identity(output)
            self.assertEqual(artifact["sha256"], artifact_fingerprint(output)["sha256"])
            run_config = {
                "schema_version": 3,
                "artifact_type": "framework_teacher_adapter",
                "role": "framework_teacher",
                "run_id": "test-run-id",
                "data_sha256": "data-sha",
                "purity_audit": {"valid": 1, "invalid": 0},
                "adapter_artifact": artifact,
                "adapter_artifact_sha256": artifact["sha256"],
            }
            _write_run_metadata(output, run_config)

            run_config_path = output / "run_config.json"
            marker = json.loads((output / "RUN_COMPLETE").read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "complete")
            self.assertEqual(marker["role"], "framework_teacher")
            self.assertEqual(marker["artifact_type"], "framework_teacher_adapter")
            self.assertEqual(marker["run_id"], "test-run-id")
            self.assertEqual(marker["adapter_artifact_sha256"], artifact["sha256"])
            self.assertEqual(
                marker["run_config_sha256"],
                hashlib.sha256(run_config_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(verify_teacher_artifact(output)["adapter_artifact"], artifact)

            weights.write_bytes(b"adapter-v2")
            with self.assertRaisesRegex(ValueError, "adapter files"):
                verify_teacher_artifact(output)

    def test_base_model_identity_hashes_metadata_but_only_stats_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text('{"model_type": "qwen"}', encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"pretend-large-weight")
            identity = lightweight_model_identity(model)

            self.assertIn("sha256", identity["metadata"]["config.json"])
            self.assertNotIn("sha256", identity["weights"]["model.safetensors"])
            self.assertEqual(identity["weights"]["model.safetensors"]["size"], 20)
            self.assertEqual(len(identity["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
