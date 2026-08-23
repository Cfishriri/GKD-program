import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_EVALUATION_RUNTIME = all(
    importlib.util.find_spec(package) is not None
    for package in ("matplotlib", "peft", "torch", "transformers")
)
if HAS_EVALUATION_RUNTIME:
    from evaluate_comparison import (
        canonicalize_predictions,
        finalize_manifest,
        load_adapter_metadata,
        load_framework_cache,
        load_prediction_rows,
        model_identity,
        pending_records,
        plot_accuracy_vs_cost,
        plot_grouped_accuracy,
        plot_paired_deltas,
        plot_paired_outcomes,
        resolve_project_paths,
        sha256_text,
        stable_framework_id,
        validate_config,
        validate_resume_provenance,
        validate_training_metadata,
        verify_completed_output_integrity,
    )
from framework_opd.evaluation import (
    artifact_fingerprint,
    experiment_signature,
    exact_mcnemar_pvalue,
    extract_final_answer,
    extract_relaxed_answer,
    file_sha256,
    paired_comparison,
    paired_interaction,
    score_prediction,
    summarize,
    validate_resume_identity,
)
from framework_opd.prompts import format_vanilla_student_prompt


class EvaluationTest(unittest.TestCase):
    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_evaluation_project_paths_resolve_from_repo_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "relocated-project"
            config = resolve_project_paths(
                {
                    "student_model": "/models/student",
                    "teacher_model": "/models/teacher",
                    "dataset": "/datasets/test.parquet",
                    "framework_teacher_adapter": "outputs/teacher",
                    "output_dir": "outputs/evaluation",
                    "adapters": {
                        "vanilla": "outputs/vanilla/student_adapter",
                        "guided": "outputs/guided/student_adapter",
                    },
                },
                root,
            )
            self.assertEqual(config["dataset"], "/datasets/test.parquet")
            self.assertEqual(config["framework_teacher_adapter"], str(root / "outputs/teacher"))
            self.assertEqual(config["output_dir"], str(root / "outputs/evaluation"))
            self.assertEqual(
                config["adapters"]["guided"],
                str(root / "outputs/guided/student_adapter"),
            )

    def test_extracts_only_marked_answer_and_normalizes_currency(self):
        self.assertEqual(extract_final_answer("Result:\n#### $1,234.50"), "1234.5")
        self.assertIsNone(extract_final_answer("unfinished: 2 + 3"))
        self.assertIsNone(extract_final_answer("the marker #### 5 is embedded in prose"))

    def test_strict_answer_requires_last_nonempty_physical_line(self):
        self.assertEqual(extract_final_answer("work\n#### 1,234\n\t\n"), "1234")
        self.assertIsNone(extract_final_answer("#### 7\npostscript"))
        self.assertIsNone(extract_final_answer("####\n7"))
        self.assertIsNone(extract_final_answer("work\n#### 7 extra"))
        self.assertIsNone(extract_final_answer("work\n#### 7\nanswer: 7"))

    def test_strict_answer_enforces_real_thousands_grouping(self):
        for valid in ("#### 1234", "#### 1,234", "#### -$12,345.60"):
            self.assertIsNotNone(extract_final_answer(valid), valid)
        for invalid in ("#### 12,34", "#### 1,23,456", "#### 1234,567", "#### 1,000,00"):
            self.assertIsNone(extract_final_answer(invalid), invalid)

    def test_relaxed_answer_falls_back_to_last_number(self):
        self.assertEqual(extract_relaxed_answer("First 4, finally 9."), "9")

    def test_strict_and_relaxed_scores_are_separate(self):
        result = score_prediction("unfinished: 2 + 3", "work\n#### 3")
        self.assertFalse(result["correct"])
        self.assertTrue(result["relaxed_correct"])
        self.assertFalse(result["has_answer_marker"])

    def test_exact_numeric_match(self):
        result = score_prediction("#### 5.0", "work\n#### 5")
        self.assertTrue(result["correct"])
        self.assertTrue(result["has_answer_marker"])

    def test_summary_reports_accuracy_and_generation_diagnostics(self):
        summary = summarize([
            {
                "correct": True,
                "relaxed_correct": True,
                "has_answer_marker": True,
                "ended_with_eos": True,
                "hit_max_tokens": False,
                "framework_failure": False,
                "generated_tokens": 10,
            },
            {
                "correct": False,
                "relaxed_correct": True,
                "has_answer_marker": False,
                "ended_with_eos": False,
                "hit_max_tokens": True,
                "framework_failure": True,
                "generated_tokens": 20,
            },
        ])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["relaxed_accuracy"], 1.0)
        self.assertEqual(summary["answer_format_rate"], 0.5)
        self.assertEqual(summary["eos_rate"], 0.5)
        self.assertEqual(summary["truncation_rate"], 0.5)
        self.assertEqual(summary["framework_failure_rate"], 0.5)
        self.assertEqual(summary["shared_framework_failure_rate"], 0.5)
        self.assertEqual(summary["average_generated_tokens"], 15)
        self.assertLess(summary["accuracy_ci95_low"], summary["accuracy"])
        self.assertGreater(summary["accuracy_ci95_high"], summary["accuracy"])

    def test_summary_separates_actual_and_shared_framework_failure_and_cost(self):
        rows = [
            {
                "correct": True,
                "has_answer_marker": True,
                "framework_failure": False,
                "shared_framework_failure": True,
                "student_prompt_tokens": 10,
                "student_output_tokens": 4,
                "framework_prompt_tokens": 0,
                "framework_output_tokens": 0,
                "token_cost_proxy": 14,
                "framework_4b_calls": 0,
            }
        ]
        summary = summarize(rows)
        self.assertEqual(summary["framework_failure_rate"], 0.0)
        self.assertEqual(summary["shared_framework_failure_rate"], 1.0)
        self.assertEqual(summary["average_token_cost_proxy"], 14)
        self.assertEqual(summary["total_framework_4b_calls"], 0)

    def test_paired_comparison_reports_discordance_delta_and_exact_test(self):
        baseline = [
            {"example_id": 0, "correct": True},
            {"example_id": 1, "correct": True},
            {"example_id": 2, "correct": False},
            {"example_id": 3, "correct": False},
        ]
        comparison = [
            {"example_id": 0, "correct": True},
            {"example_id": 1, "correct": False},
            {"example_id": 2, "correct": True},
            {"example_id": 3, "correct": True},
        ]
        result = paired_comparison(
            baseline,
            comparison,
            baseline_name="a",
            comparison_name="b",
            seed=7,
            bootstrap_samples=200,
        )
        self.assertEqual(result["accuracy_delta"], 0.25)
        self.assertEqual(result["outcomes"]["both_correct"], 1)
        self.assertEqual(result["outcomes"]["baseline_only_correct"], 1)
        self.assertEqual(result["outcomes"]["comparison_only_correct"], 2)
        self.assertEqual(result["outcomes"]["both_wrong"], 0)
        self.assertEqual(result["mcnemar_exact_pvalue"], 1.0)

    def test_paired_comparison_is_deterministic_and_requires_same_ids(self):
        rows = [{"example_id": index, "correct": index % 2 == 0} for index in range(6)]
        kwargs = {
            "baseline_name": "a",
            "comparison_name": "b",
            "seed": 11,
            "bootstrap_samples": 100,
        }
        first = paired_comparison(rows, list(reversed(rows)), **kwargs)
        second = paired_comparison(rows, list(reversed(rows)), **kwargs)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "identical example_id"):
            paired_comparison(rows, rows[:-1], **kwargs)

    def test_exact_mcnemar_known_edge_cases(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_pvalue(0, 5), 0.0625)

    def test_paired_interaction_uses_all_four_cells(self):
        def rows(values):
            return [{"example_id": index, "correct": value} for index, value in enumerate(values)]

        result = paired_interaction(
            rows([False, False]),
            rows([False, False]),
            rows([False, True]),
            rows([True, True]),
            seed=3,
            bootstrap_samples=100,
        )
        self.assertEqual(result["accuracy_delta"], 0.5)
        self.assertIsNone(result["outcomes"])
        self.assertIsNone(result["mcnemar_exact_pvalue"])

    def test_artifact_fingerprint_hashes_adapter_weights_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            fingerprint = artifact_fingerprint(adapter)
            self.assertEqual(fingerprint["type"], "directory")
            self.assertEqual(
                {entry["relative_path"] for entry in fingerprint["files"]},
                {"adapter_config.json", "adapter_model.safetensors"},
            )
            self.assertEqual(len(fingerprint["sha256"]), 64)

    def test_experiment_signature_ignores_only_resume_switch(self):
        base = {"seed": 7, "resume": False}
        self.assertEqual(experiment_signature(base), experiment_signature({**base, "resume": True}))
        self.assertNotEqual(experiment_signature(base), experiment_signature({**base, "seed": 8}))

    def test_resume_identity_rejects_config_or_artifact_changes(self):
        manifest = {"experiment_signature": "config", "artifact_fingerprints": {"data": "hash"}}
        validate_resume_identity(manifest, "config", {"data": "hash"})
        with self.assertRaisesRegex(ValueError, "config"):
            validate_resume_identity(manifest, "changed", {"data": "hash"})
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            validate_resume_identity(manifest, "config", {"data": "changed"})

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_resume_provenance_is_immutable(self):
        manifest = {"experiment_signature": "signature", "provenance": {"runtime": "locked"}}
        validate_resume_provenance(manifest, "signature", {"runtime": "locked"})
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_resume_provenance(manifest, "signature", {"runtime": "changed"})

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_model_identity_hashes_config_tokenizer_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            identity = model_identity(model)
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertEqual(
                {entry["relative_path"] for entry in identity["files"]},
                {"config.json", "tokenizer_config.json", "model.safetensors.index.json"},
            )
            self.assertEqual(identity["weight_files"][0]["size_bytes"], 7)
            self.assertGreater(identity["weight_files"][0]["mtime_ns"], 0)

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_training_provenance_rejects_same_adapter_and_wrong_beta(self):
        from train_opd import _resume_signature
        from train_teacher import lightweight_model_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def adapter(name, weight):
                path = root / name
                path.mkdir()
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_bytes(weight)
                return path

            def model(name):
                path = root / name
                path.mkdir()
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
                (path / "model.safetensors").write_bytes(name.encode())
                return path

            vanilla_path = adapter("vanilla", b"vanilla")
            guided_path = adapter("guided", b"guided")
            framework_path = adapter("framework", b"framework")
            student_model = model("student")
            teacher_model = model("teacher")
            dataset = root / "train.parquet"
            dataset.write_bytes(b"training records")
            framework_data = root / "frameworks.jsonl"
            framework_data.write_bytes(b"framework records")
            framework_data_sha = file_sha256(framework_data)
            generation_audit_path = root / "generation-audit.json"
            generation_audit_payload = {
                "status": "complete",
                "requested_valid": 10,
                "valid": 10,
                "output_sha256": framework_data_sha,
            }
            generation_audit_path.write_text(
                json.dumps(generation_audit_payload), encoding="utf-8"
            )
            common = {
                "role": "opd_student",
                "student_model": str(student_model),
                "teacher_model": str(teacher_model),
                "dataset": str(dataset),
                "beta": 1.0,
                "max_steps": 10,
            }
            vanilla_config = {
                **common,
                "mode": "vanilla",
                "run_id": "v",
                "output_dir": "v",
            }
            guided_config = {
                **common,
                "mode": "guided",
                "run_id": "g",
                "output_dir": "g",
                "framework_teacher_adapter": str(framework_path),
            }
            vanilla_signature = _resume_signature(vanilla_config, "vanilla")
            guided_signature = _resume_signature(guided_config, "guided")
            metadata = {
                "vanilla": {
                    "adapter": artifact_fingerprint(vanilla_path),
                    "run_config": vanilla_config,
                    "run_config_sha256": "v-config",
                    "completion": {"role": "opd_student", "run_id": "v"},
                    "run_manifest": {
                        "schema_version": 2,
                        "role": "opd_student",
                        "run_id": "v",
                        "status": "complete",
                        "run_config_sha256": "v-config",
                        "adapter_artifact_sha256": artifact_fingerprint(vanilla_path)["sha256"],
                        "run_signature": vanilla_signature,
                    },
                },
                "guided": {
                    "adapter": artifact_fingerprint(guided_path),
                    "run_config": guided_config,
                    "run_config_sha256": "g-config",
                    "completion": {"role": "opd_student", "run_id": "g"},
                    "run_manifest": {
                        "schema_version": 2,
                        "role": "opd_student",
                        "run_id": "g",
                        "status": "complete",
                        "run_config_sha256": "g-config",
                        "adapter_artifact_sha256": artifact_fingerprint(guided_path)["sha256"],
                        "run_signature": guided_signature,
                    },
                },
                "framework_teacher": {
                    "adapter": artifact_fingerprint(framework_path),
                    "run_config": {
                        "role": "framework_teacher",
                        "artifact_type": "framework_teacher_adapter",
                        "base_model": str(teacher_model),
                        "base_model_identity": lightweight_model_identity(teacher_model),
                        "run_id": "framework-run",
                        "data": str(framework_data),
                        "data_sha256": framework_data_sha,
                        "generation_audit": {
                            "path": str(generation_audit_path),
                            "sha256": file_sha256(generation_audit_path),
                            **generation_audit_payload,
                        },
                        "expected_records": 10,
                        "num_records": 10,
                        "purity_audit": {
                            "total": 10,
                            "valid": 10,
                            "invalid": 0,
                            "leakage_records": 0,
                        },
                    },
                    "completion": {
                        "role": "framework_teacher",
                        "artifact_type": "framework_teacher_adapter",
                        "run_id": "framework-run",
                    },
                },
            }
            config = {
                "student_model": str(student_model),
                "teacher_model": str(teacher_model),
                "framework_teacher_expected_records": 10,
            }
            validate_training_metadata(config, metadata)
            metadata["framework_teacher"]["run_config"]["purity_audit"]["invalid"] = 1
            with self.assertRaisesRegex(ValueError, "purity"):
                validate_training_metadata(config, metadata)
            metadata["framework_teacher"]["run_config"]["purity_audit"]["invalid"] = 0
            metadata["guided"]["run_config"]["beta"] = 0.5
            with self.assertRaisesRegex(ValueError, "beta"):
                validate_training_metadata(config, metadata)
            metadata["guided"]["run_config"]["beta"] = 1.0
            metadata["guided"]["adapter"] = metadata["vanilla"]["adapter"]
            with self.assertRaisesRegex(ValueError, "same fingerprint"):
                validate_training_metadata(config, metadata)
            metadata["guided"]["adapter"] = artifact_fingerprint(guided_path)
            metadata["guided"]["run_manifest"]["run_signature"] = vanilla_signature
            with self.assertRaisesRegex(ValueError, "run_signature"):
                validate_training_metadata(config, metadata)

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_adapter_completion_marker_binds_run_config_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            run_config = adapter / "run_config.json"
            run_config.write_text('{"role":"framework_teacher"}\n', encoding="utf-8")
            fingerprint = artifact_fingerprint(adapter)
            completion = {
                "status": "complete",
                "run_config_sha256": file_sha256(run_config),
                "adapter_artifact_sha256": fingerprint["sha256"],
            }
            (adapter / "RUN_COMPLETE").write_text(json.dumps(completion), encoding="utf-8")
            loaded = load_adapter_metadata(adapter)
            self.assertEqual(loaded["adapter"]["sha256"], fingerprint["sha256"])
            completion["adapter_artifact_sha256"] = "wrong"
            (adapter / "RUN_COMPLETE").write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weights"):
                load_adapter_metadata(adapter)

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_framework_cache_and_partial_cell_resume_preserve_499_rows(self):
        framework = [
            "Identify the requested relationship",
            "Represent the relationship symbolically",
            "Carry out the required reasoning",
            "Check and present the conclusion",
        ]
        adapter_sha = "a" * 64
        records = [
            {"example_id": index, "question": f"Question {index}?", "answer": "#### 1"}
            for index in range(500)
        ]
        cache = {}
        for record in records:
            framework_id = stable_framework_id(
                record["example_id"], record["question"], framework, adapter_sha
            )
            cache[record["example_id"]] = {
                "framework_id": framework_id,
                "example_id": record["example_id"],
                "question": record["question"],
                "framework": framework,
                "framework_valid": True,
                "framework_failure": False,
                "framework_fallback": False,
                "framework_attempts": 1,
                "framework_validation_errors": [],
                "framework_prompt_tokens": 8,
                "framework_output_tokens": 4,
                "framework_latency_seconds": 0.1,
                "framework_hit_max_attempts": 0,
                "framework_last_ended_with_eos": True,
                "framework_closed_tag": True,
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "framework_cache.jsonl"
            cache_path.write_text(
                "".join(json.dumps(cache[index]) + "\n" for index in range(500)),
                encoding="utf-8",
            )
            loaded_cache = load_framework_cache(cache_path, records, adapter_sha)
            self.assertEqual(len(loaded_cache), 500)

            rows = []
            for record in records[:499]:
                prompt = format_vanilla_student_prompt(record["question"])
                row = {
                    "cell": "vanilla_no_framework",
                    "adapter": "vanilla",
                    "framework_condition": "no_framework",
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "reference": record["answer"],
                    "framework": [],
                    "framework_id": None,
                    "shared_framework_id": cache[record["example_id"]]["framework_id"],
                    "framework_used": False,
                    "framework_failure": False,
                    "shared_framework_failure": False,
                    "shared_framework_attempts": 1,
                    "shared_framework_hit_max_attempts": 0,
                    "shared_framework_closed_tag": True,
                    "prediction": "#### 1",
                    "student_prompt_sha256": sha256_text(prompt),
                    "student_prompt_tokens": 5,
                    "student_output_tokens": 1,
                    "student_latency_seconds": 0.01,
                    "framework_prompt_tokens": 0,
                    "framework_output_tokens": 0,
                    "framework_latency_seconds": 0.0,
                    "framework_4b_calls": 0,
                    "token_cost_proxy": 6,
                    "generated_tokens": 1,
                    "ended_with_eos": True,
                    "hit_max_tokens": False,
                    **score_prediction("#### 1", record["answer"]),
                }
                rows.append(row)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded = load_prediction_rows(
                predictions,
                records,
                loaded_cache,
                ["vanilla_no_framework"],
            )
            self.assertEqual(len(loaded["vanilla_no_framework"]), 499)
            missing = pending_records(records, loaded["vanilla_no_framework"])
            self.assertEqual([row["example_id"] for row in missing], [499])

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_plots_and_final_output_inventory_are_created(self):
        cells = [
            "vanilla_no_framework",
            "guided_no_framework",
            "vanilla_with_framework",
            "guided_with_framework",
        ]
        summaries = [
            {
                "cell": cell,
                "total": 10,
                "accuracy": 0.5 + index * 0.05,
                "accuracy_ci95_low": 0.3,
                "accuracy_ci95_high": 0.8,
                "average_token_cost_proxy": 20 + index * 5,
                "total_framework_4b_calls": 10 if cell.endswith("with_framework") else 0,
                "total_framework_latency_seconds": 2.0 if cell.endswith("with_framework") else 0.0,
            }
            for index, cell in enumerate(cells)
        ]
        comparisons = [
            {
                "name": "guided_minus_vanilla_no_framework",
                "analysis_tier": "primary",
                "accuracy_delta": 0.1,
                "bootstrap_ci95_low": -0.01,
                "bootstrap_ci95_high": 0.2,
                "outcomes": {
                    "both_correct": 3,
                    "baseline_only_correct": 1,
                    "comparison_only_correct": 2,
                    "both_wrong": 4,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot_grouped_accuracy(root, summaries, include_base=False)
            plot_paired_deltas(root, comparisons)
            plot_paired_outcomes(root, comparisons)
            plot_accuracy_vs_cost(root, summaries)
            for name in (
                "grouped_accuracy.png",
                "paired_deltas.png",
                "paired_outcomes.png",
                "accuracy_vs_cost.png",
            ):
                self.assertGreater((root / name).stat().st_size, 0)
            manifest = {"status": "running"}
            finalize_manifest(root, manifest)
            stored = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("paired_outcomes.png", stored["outputs"])
            self.assertEqual(len(stored["outputs"]["paired_outcomes.png"]["sha256"]), 64)
            verify_completed_output_integrity(root, stored)
            (root / "paired_outcomes.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                verify_completed_output_integrity(root, stored)

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_prediction_canonicalization_uses_cell_then_record_order(self):
        records = [{"example_id": 2}, {"example_id": 1}]
        rows_by_cell = {
            "first": [{"example_id": 1, "value": "b"}, {"example_id": 2, "value": "a"}],
            "second": [{"example_id": 1, "value": "d"}, {"example_id": 2, "value": "c"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            canonicalize_predictions(path, rows_by_cell, ["first", "second"], records)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [(row["value"], row["example_id"]) for row in rows],
                [("a", 2), ("b", 1), ("c", 2), ("d", 1)],
            )

    @unittest.skipUnless(HAS_EVALUATION_RUNTIME, "evaluation runtime dependencies are unavailable")
    def test_config_validation_runs_before_generation(self):
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_config({})

    def test_evaluation_config_defines_both_adapters_and_paired_bootstrap(self):
        config_path = Path(__file__).parents[1] / "configs" / "evaluation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(set(config["adapters"]), {"vanilla", "guided"})
        self.assertIs(config["resume"], False)
        self.assertGreater(config["bootstrap_samples"], 0)
        self.assertGreater(config["framework_max_attempts"], 0)
        self.assertEqual(config["framework_teacher_expected_records"], 1000)


if __name__ == "__main__":
    unittest.main()
