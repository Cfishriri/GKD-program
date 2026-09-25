import unittest
from unittest.mock import patch

import torch

from framework_opd.eval_batching import (
    generate_batch,
    generate_framework_batch,
    iter_parallel_batches,
)
from framework_opd.rollout import FALLBACK_FRAMEWORK, GenerationResult


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, prompts, **kwargs):
        width = max(map(len, prompts))
        return {
            "input_ids": torch.tensor([[0] * (width - len(p)) + list(map(ord, p)) for p in prompts]),
            "attention_mask": torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts]),
        }

    def batch_decode(self, rows, **kwargs):
        return ["".join(chr(int(i)) for i in row if i) for row in rows]

    def decode(self, row, **kwargs):
        return self.batch_decode([row])[0]


class ScriptedModel:
    device = torch.device("cpu")

    def __init__(self, completions):
        self.completions = [list(map(ord, text)) for text in completions]

    def generate(self, input_ids, max_new_tokens, stopping_criteria, **kwargs):
        done = torch.zeros(len(input_ids), dtype=torch.bool)
        for step in range(max_new_tokens):
            tokens = [0 if done[i] or step >= len(row) else row[step] for i, row in enumerate(self.completions)]
            input_ids = torch.cat([input_ids, torch.tensor(tokens)[:, None]], dim=1)
            done |= stopping_criteria(input_ids, None)
            done |= torch.tensor(tokens) == 0
            if done.all():
                break
        return input_ids


class EvalBatchingTest(unittest.TestCase):
    def test_resume_entrypoint_rejects_old_schema_changed_batch_and_source_without_writes(self):
        import json
        import tempfile
        from pathlib import Path
        import evaluate_comparison as evaluation
        from framework_opd.evaluation import experiment_signature
        config = json.loads((Path(__file__).parents[1] / "configs/evaluation.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config["output_dir"] = str(root / "out")
            Path(config["output_dir"]).mkdir()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            signature = experiment_signature(evaluation.resolve_project_paths(config))
            changed = dict(config, student_batch_size=config["student_batch_size"] + 1)
            provenance = {"source": {"source_sha256": "current"}}
            cases = [
                {"schema_version": 5, "experiment_signature": signature, "provenance": provenance},
                {"schema_version": evaluation.SCHEMA_VERSION,
                 "experiment_signature": experiment_signature(evaluation.resolve_project_paths(changed)), "provenance": provenance},
                {"schema_version": evaluation.SCHEMA_VERSION, "experiment_signature": signature,
                 "provenance": {"source": {"source_sha256": "old"}}},
            ]
            for manifest in cases:
                manifest["status"] = "running"
                path = Path(config["output_dir"]) / "run_manifest.json"
                path.write_text(json.dumps(manifest))
                before = path.read_bytes()
                with self.subTest(manifest=manifest), patch("sys.argv", ["evaluate_comparison.py", "--config", str(config_path), "--resume"]), \
                    patch.object(evaluation, "collect_provenance", return_value=provenance), \
                    patch.object(evaluation, "run_evaluation") as run, \
                    patch.object(evaluation, "write_json") as write, self.assertRaises(ValueError):
                    evaluation.main()
                run.assert_not_called()
                write.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_evaluation_batch_config_validation(self):
        import json
        from pathlib import Path
        from evaluate_comparison import validate_config
        config = json.loads((Path(__file__).parents[1] / "configs/evaluation.json").read_text())
        validate_config(config)
        for change in (
            {"student_batch_size": 0}, {"framework_batch_size": True},
            {"eval_devices": []}, {"eval_devices": ["cuda:0", "cuda:00"]},
            {"eval_devices": ["cuda"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(dict(config, **change))

    def test_per_row_answer_and_eos_stopping_excludes_padding(self):
        results = generate_batch(
            ScriptedModel(["#### 20\nignored", "abc\0", "no answer here forever"]),
            CharacterTokenizer(), ["a", "long prompt", "xyz"], 12,
        )
        self.assertEqual([r.prompt_tokens for r in results], [1, 11, 3])
        self.assertEqual(results[0].text, "#### 20")
        self.assertEqual(len(results[0].token_ids), len("#### 20\n"))
        self.assertTrue(results[0].stopped_on_answer)
        self.assertFalse(results[0].ended_with_eos)
        self.assertEqual(results[1].token_ids, (97, 98, 99, 0))
        self.assertTrue(results[1].ended_with_eos)
        self.assertFalse(results[1].hit_max_tokens)
        self.assertTrue(results[2].hit_max_tokens)
        self.assertEqual(len(results[2].token_ids), 12)

    def test_incomplete_answer_does_not_stop_before_last_digit(self):
        result = generate_batch(ScriptedModel(["#### 200"]), CharacterTokenizer(), ["q"], 8)[0]
        self.assertEqual(result.text, "#### 200")
        self.assertFalse(result.stopped_on_answer)
        self.assertTrue(result.hit_max_tokens)

    def test_framework_generation_does_not_use_answer_stop(self):
        result = generate_batch(
            ScriptedModel(["#### 1\nkeep going"]), CharacterTokenizer(), ["q"], 12,
            stop_on_answer=False,
        )[0]
        self.assertEqual(len(result.token_ids), 12)
        self.assertEqual(result.text, "#### 1")
        self.assertTrue(result.stopped_on_answer)

    def test_framework_answer_leakage_before_closing_tag_is_retried(self):
        text = "1. Identify quantities.\n2. Relate quantities.\n#### 42\n</framework>"
        result = generate_framework_batch(ScriptedModel([text]), CharacterTokenizer(),
            [{"question": "q", "answer": "#### 42"}],
            max_new_tokens=128, temperature=0.0, max_attempts=2)[0]
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.attempts, 2)
        self.assertIn("framework_not_closed", result.validation_errors)

    def test_sampled_generation_retains_original_top_p(self):
        model = ScriptedModel(["abc"])
        with patch.object(model, "generate", wraps=model.generate) as mocked:
            generate_batch(model, CharacterTokenizer(), ["q"], 4, temperature=0.4)
        self.assertEqual(mocked.call_args.kwargs["top_p"], 0.9)
        self.assertEqual(mocked.call_args.kwargs["temperature"], 0.4)

    def test_framework_retries_only_failures_and_accounts_all_attempts(self):
        valid = "\n".join(f"{i}. {step}" for i, step in enumerate(FALLBACK_FRAMEWORK, 1)) + "\n</framework>"
        def gen(text):
            return GenerationResult((1, 2), text, True, False, prompt_tokens=7)
        with patch("framework_opd.eval_batching.generate_batch", side_effect=[
            [gen(valid), gen("incomplete")], [gen(valid)],
        ]) as mock:
            results = generate_framework_batch(None, None, [
                {"question": "question one", "answer": "#### 42"},
                {"question": "question two", "answer": "#### 42"},
            ], max_new_tokens=128, temperature=0.0, max_attempts=3)
        self.assertEqual([r.attempts for r in results], [1, 2])
        self.assertEqual([r.prompt_tokens for r in results], [7, 14])
        self.assertEqual([r.generated_tokens for r in results], [2, 4])
        self.assertFalse(any(r.used_fallback for r in results))
        self.assertEqual(len(mock.call_args_list[1].args[2]), 1)
        self.assertNotIn("42", mock.call_args_list[0].args[2][0])

    def test_exhausted_framework_uses_fallback(self):
        invalid = GenerationResult((1,), "incomplete", False, True, prompt_tokens=3)
        with patch("framework_opd.eval_batching.generate_batch", return_value=[invalid]):
            result = generate_framework_batch(None, None, [{"question": "q", "answer": "#### 9"}],
                max_new_tokens=1, temperature=0.0, max_attempts=2)[0]
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.hit_max_attempts, 2)

    def test_parallel_batches_preserve_order_and_use_both_replicas(self):
        import threading
        barrier = threading.Barrier(2)
        def generate(model, tokenizer, records):
            barrier.wait(timeout=5)
            return [(model, r) for r in records]
        rows = list(iter_parallel_batches(list(range(7)), [("gpu0", None), ("gpu1", None)], 2, generate))
        self.assertEqual([row[0] for row in rows], list(range(7)))
        self.assertEqual([row[1][0] for row in rows], ["gpu0", "gpu0", "gpu1", "gpu1", "gpu0", "gpu0", "gpu1"])
        self.assertTrue(all(row[2] >= 0 for row in rows))

    def test_invalid_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_parallel_batches([1], [(None, None)], 0, None))
