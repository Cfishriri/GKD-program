import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import train_opd


class TrainingResumeTests(unittest.TestCase):
    @staticmethod
    def _write_adapter(path: Path, payload: bytes = b"adapter") -> None:
        path.mkdir(parents=True, exist_ok=False)
        (path / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(payload)

    def _signature_fixture(self, root: Path) -> dict:
        dataset = root / "dataset.jsonl"
        dataset.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")

        student = root / "student"
        teacher = root / "teacher"
        for model, size in ((student, 3), (teacher, 5)):
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "test", "hidden_size": size}),
                encoding="utf-8",
            )
            (model / "tokenizer_config.json").write_text(
                json.dumps({"eos_token": "<eos>"}),
                encoding="utf-8",
            )
            (model / "model.safetensors").write_bytes(b"weight")

        adapter = root / "framework-adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")

        project = root / "project"
        for relative in train_opd.SIGNATURE_SOURCE_FILES:
            source = project / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"source:{relative}\n", encoding="utf-8")

        return {
            "mode": "guided",
            "student_model": str(student),
            "teacher_model": str(teacher),
            "framework_teacher_adapter": str(adapter),
            "dataset": str(dataset),
            "seed": 42,
            "max_steps": 4,
            "beta": 1.0,
            "temperature": 1.0,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.0001,
            "lora_r": 8,
            "lora_alpha": 16,
            "framework_max_new_tokens": 32,
            "framework_max_attempts": 3,
            "solution_max_new_tokens": 64,
            "generation_temperature": 0.7,
            "student_device": "cuda:0",
            "teacher_device": "cuda:1",
            "output_dir": str(root / "output"),
            "project_root": project,
        }

    def test_content_changes_signature_but_device_topology_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._signature_fixture(Path(temporary))
            project = fixture.pop("project_root")
            original = train_opd._resume_signature(fixture, "guided", project)

            moved_devices = dict(fixture, student_device="cuda:7", teacher_device="cuda:3")
            self.assertEqual(
                original,
                train_opd._resume_signature(moved_devices, "guided", project),
            )

            Path(fixture["dataset"]).write_text(
                '{"question":"changed","answer":"a"}\n',
                encoding="utf-8",
            )
            changed_dataset = train_opd._resume_signature(fixture, "guided", project)
            self.assertNotEqual(original["sha256"], changed_dataset["sha256"])

            Path(fixture["dataset"]).write_text(
                '{"question":"q","answer":"a"}\n',
                encoding="utf-8",
            )
            Path(fixture["framework_teacher_adapter"], "adapter_model.safetensors").write_bytes(
                b"adapter-v2"
            )
            changed_adapter = train_opd._resume_signature(fixture, "guided", project)
            self.assertNotEqual(original["sha256"], changed_adapter["sha256"])

    def test_resume_checkpoint_must_be_in_configured_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_output = root / "first"
            second_output = root / "second"
            checkpoint = first_output / "checkpoints" / "checkpoint-000001"
            checkpoint.mkdir(parents=True)
            second_output.mkdir()

            with self.assertRaisesRegex(ValueError, "directly inside"):
                train_opd._validate_resume_checkpoint_location(checkpoint, second_output)

    def test_checkpoint_and_metrics_run_id_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "CHECKPOINT_COMPLETE").write_text(
                json.dumps({"run_id": "foreign", "rollout_step": 1}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "marker run_id mismatch"):
                train_opd._load_training_state(checkpoint, {"sha256": "sig"}, "expected")

            metrics = root / "metrics.jsonl"
            metrics.write_text(
                json.dumps({"run_id": "foreign", "rollout_step": 1}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metrics run_id mismatch"):
                train_opd._rewind_metrics_for_resume(metrics, 1, "expected")

    def test_metrics_rewind_preserves_provenance_and_backs_up_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.jsonl"
            metrics.write_text(
                "".join(
                    json.dumps({"run_id": "run", "rollout_step": step}) + "\n"
                    for step in (1, 2, 3)
                ),
                encoding="utf-8",
            )
            train_opd._rewind_metrics_for_resume(metrics, 2, "run")

            kept = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["rollout_step"] for record in kept], [1, 2])
            backup = root / "metrics.before-resume-000002.jsonl"
            self.assertTrue(backup.is_file())
            self.assertEqual(len(backup.read_text(encoding="utf-8").splitlines()), 3)

    def test_manifest_identity_survives_resume_and_completion_marker_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = {
                "output_dir": str(output),
                "student_device": "cuda:0",
                "teacher_device": "cuda:1",
            }
            signature = {"sha256": "signature", "payload": {"dataset": "fixed"}}
            with patch.object(
                train_opd,
                "_runtime_snapshot",
                side_effect=[{"timestamp": "initial"}, {"timestamp": "resume"}],
            ):
                initial = train_opd._initialize_new_run(output, config, "vanilla", signature)
                checkpoint = output / "checkpoints" / "checkpoint-000001"
                checkpoint.mkdir(parents=True)
                resumed = train_opd._append_resume_history(
                    output,
                    initial,
                    config,
                    checkpoint,
                    1,
                )
            self._write_adapter(output / "student_adapter")
            completed = train_opd._mark_run_complete(
                output,
                initial["run_id"],
                rollout_step=4,
                optimizer_step=2,
            )

            self.assertEqual(completed["run_id"], initial["run_id"])
            self.assertEqual(completed["run_signature"], signature)
            self.assertEqual(completed["initial_runtime"], {"timestamp": "initial"})
            self.assertEqual(resumed["resume_history"][0]["runtime"], {"timestamp": "resume"})
            marker = json.loads((output / "RUN_COMPLETE").read_text(encoding="utf-8"))
            self.assertEqual(marker["run_id"], initial["run_id"])
            self.assertEqual(marker["role"], "opd_student")
            self.assertEqual(marker["status"], "complete")
            self.assertEqual(
                marker["run_config_sha256"],
                train_opd._sha256_file(output / "run_config.json"),
            )
            self.assertEqual(marker["rollout_step"], 4)
            self.assertEqual(
                marker["adapter_artifact_sha256"],
                train_opd._adapter_artifact_sha256(output / "student_adapter"),
            )

            # Repeating finalization is a no-op when all provenance agrees.
            repeated = train_opd._mark_run_complete(
                output,
                initial["run_id"],
                rollout_step=4,
                optimizer_step=2,
            )
            self.assertEqual(repeated["completed_at"], completed["completed_at"])

            # A crash after the complete manifest but before marker publication is repairable.
            (output / "RUN_COMPLETE").unlink()
            repaired = train_opd._mark_run_complete(
                output,
                initial["run_id"],
                rollout_step=4,
                optimizer_step=2,
            )
            self.assertEqual(repaired["completed_at"], completed["completed_at"])
            self.assertTrue((output / "RUN_COMPLETE").is_file())

    def test_completed_manifest_recovery_requires_the_final_checkpoint(self):
        manifest = {
            "status": "complete",
            "final_rollout_step": 8,
            "final_optimizer_step": 2,
        }
        train_opd._validate_completion_recovery(
            manifest,
            {"rollout_step": 8, "optimizer_step": 2},
            8,
        )
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            train_opd._validate_completion_recovery(
                manifest,
                {"rollout_step": 7, "optimizer_step": 2},
                8,
            )

    def test_metrics_backup_survives_failure_before_authoritative_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.jsonl"
            original = "".join(
                json.dumps({"run_id": "run", "rollout_step": step}) + "\n"
                for step in (1, 2, 3)
            )
            metrics.write_text(original, encoding="utf-8")
            real_atomic_write = train_opd._atomic_write_text
            calls = 0

            def fail_second_write(path, text):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected authoritative replace failure")
                return real_atomic_write(path, text)

            with patch.object(train_opd, "_atomic_write_text", side_effect=fail_second_write):
                with self.assertRaisesRegex(OSError, "injected"):
                    train_opd._rewind_metrics_for_resume(metrics, 2, "run")

            self.assertEqual(metrics.read_text(encoding="utf-8"), original)
            self.assertEqual(
                (root / "metrics.before-resume-000002.jsonl").read_text(encoding="utf-8"),
                original,
            )

    def test_adapter_publication_is_atomic_and_preserves_previous_evidence(self):
        class FakeStudent:
            def save_pretrained(self, path):
                TrainingResumeTests._write_adapter(Path(path), b"new")

        class FakeTokenizer:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self._write_adapter(output / "student_adapter", b"old")
            result = train_opd._publish_student_adapter(
                FakeStudent(),
                FakeTokenizer(),
                output,
                allow_existing=True,
            )
            self.assertEqual((output / "student_adapter" / "adapter_model.safetensors").read_bytes(), b"new")
            evidence = Path(result["preserved_previous_path"])
            self.assertTrue(evidence.is_dir())
            self.assertEqual((evidence / "adapter_model.safetensors").read_bytes(), b"old")
            self.assertEqual(
                result["sha256"],
                train_opd._adapter_artifact_sha256(output / "student_adapter"),
            )

    def test_adapter_hash_matches_evaluation_fingerprint(self):
        from framework_opd.evaluation import artifact_fingerprint

        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            self._write_adapter(adapter, b"shared-fingerprint")
            self.assertEqual(
                train_opd._adapter_artifact_sha256(adapter),
                artifact_fingerprint(adapter)["sha256"],
            )

    def test_adapter_save_failure_does_not_touch_existing_final_adapter(self):
        class FakeStudent:
            def save_pretrained(self, path):
                TrainingResumeTests._write_adapter(Path(path), b"partial-new")

        class FailingTokenizer:
            def save_pretrained(self, path):
                raise OSError("injected tokenizer save failure")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self._write_adapter(output / "student_adapter", b"old")
            with self.assertRaisesRegex(OSError, "injected"):
                train_opd._publish_student_adapter(
                    FakeStudent(),
                    FailingTokenizer(),
                    output,
                    allow_existing=True,
                )
            self.assertEqual((output / "student_adapter" / "adapter_model.safetensors").read_bytes(), b"old")
            self.assertTrue(list(output.glob(".student_adapter.tmp.*")))

    def test_device_topology_allows_only_card_number_migration(self):
        saved = train_opd._device_topology(
            {"student": "cuda:0", "teacher": "cuda:1"}
        )
        train_opd._validate_device_topology(
            saved,
            {"student": "cuda:4", "teacher": "cuda:7"},
        )
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            train_opd._validate_device_topology(
                saved,
                {"student": "cuda:4", "teacher": "cuda:4"},
            )
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            train_opd._validate_device_topology(
                saved,
                {"student": "cpu", "teacher": "cuda:7"},
            )

    def test_zero_steps_empty_dataset_and_reused_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_steps must be positive"):
            train_opd._validate_positive_max_steps(0)
        with self.assertRaisesRegex(ValueError, "contains no records"):
            train_opd._validate_nonempty_records([], "empty.jsonl")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "old-run").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty output"):
                train_opd._validate_new_output_lifecycle(output)

    def test_framework_rollout_diagnostics_are_preserved_in_metrics(self):
        rollout = SimpleNamespace(
            framework_attempts=3,
            framework_fallback=True,
            framework_validation_errors=["numeric_literal"],
            framework_prompt_tokens=71,
            framework_generated_tokens=29,
            framework_hit_max_attempts=2,
            framework_last_ended_with_eos=False,
            framework_closed_tag=True,
        )
        self.assertEqual(
            train_opd._framework_rollout_metrics(rollout),
            {
                "framework_attempts": 3,
                "framework_fallback": True,
                "framework_validation_errors": ["numeric_literal"],
                "framework_prompt_tokens": 71,
                "framework_generated_tokens": 29,
                "framework_hit_max_attempts": 2,
                "framework_last_ended_with_eos": False,
                "framework_closed_tag": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
