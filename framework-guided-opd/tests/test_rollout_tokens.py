import unittest
from unittest.mock import patch

import torch

from framework_opd.rollout import (
    FrameworkGenerationResult,
    GenerationResult,
    _generate_text,
    generate_framework,
    generate_framework_result,
    generate_opd_rollout,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Deliberately encode decoded completion text differently. If rollout
        # re-tokenizes it, the test will expose token 99 instead of sampled IDs.
        ids = [99] if text == "normalized completion" else [1, 2]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "normalized completion"


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, generated_ids):
        self.generated_ids = generated_ids
        self.generation_kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generation_kwargs = kwargs
        generated = torch.tensor([self.generated_ids], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


class RolloutTokenTest(unittest.TestCase):
    def test_answer_aware_generation_crops_overshoot_and_records_stop_reason(self):
        class AnswerTokenizer(FakeTokenizer):
            def decode(self, token_ids, skip_special_tokens=True):
                return "reasoning\n#### 20\ncontinued"

        model = FakeModel([5, 6])
        result = _generate_text(
            model,
            AnswerTokenizer(),
            "prompt containing #### format instructions",
            2,
            0.0,
            stop_on_answer=True,
        )

        self.assertEqual(result.text, "reasoning\n#### 20")
        self.assertEqual(result.token_ids, (5, 6))
        self.assertTrue(result.stopped_on_answer)
        self.assertFalse(result.hit_max_tokens)
        self.assertIn("stopping_criteria", model.generation_kwargs)

    def test_generation_result_reports_actual_eos(self):
        model = FakeModel([5, 9])
        result = _generate_text(model, FakeTokenizer(), "prompt", 2, 0.0)

        self.assertEqual(result.token_ids, (5, 9))
        self.assertTrue(result.ended_with_eos)
        self.assertFalse(result.hit_max_tokens)
        self.assertEqual(result.prompt_tokens, 2)
        self.assertNotIn("temperature", model.generation_kwargs)
        self.assertNotIn("top_p", model.generation_kwargs)

    def test_sampling_parameters_are_only_passed_when_sampling(self):
        model = FakeModel([5])
        _generate_text(model, FakeTokenizer(), "prompt", 2, 0.7)

        self.assertTrue(model.generation_kwargs["do_sample"])
        self.assertEqual(model.generation_kwargs["temperature"], 0.7)
        self.assertEqual(model.generation_kwargs["top_p"], 0.9)

    def test_generation_result_reports_length_truncation_without_adding_eos(self):
        result = _generate_text(FakeModel([5, 6]), FakeTokenizer(), "prompt", 2, 0.0)

        self.assertEqual(result.token_ids, (5, 6))
        self.assertFalse(result.ended_with_eos)
        self.assertTrue(result.hit_max_tokens)

    def test_rollout_uses_sampled_ids_without_decode_encode_round_trip(self):
        sampled = GenerationResult(
            token_ids=(7, 8),
            text="normalized completion",
            ended_with_eos=False,
            hit_max_tokens=True,
            stopped_on_answer=False,
        )
        with patch("framework_opd.rollout._generate_text", return_value=sampled):
            rollout = generate_opd_rollout(
                None,
                None,
                FakeTokenizer(),
                "question",
                mode="vanilla",
                framework_max_new_tokens=4,
                solution_max_new_tokens=2,
                framework_temperature=0.0,
                solution_temperature=0.0,
            )

        self.assertEqual(rollout.generated_token_ids, (7, 8))
        self.assertEqual(rollout.input_ids.tolist(), [[1, 2, 7, 8]])
        self.assertEqual(rollout.labels.tolist(), [[-100, -100, 7, 8]])
        self.assertNotIn(9, rollout.input_ids.tolist()[0])
        self.assertTrue(rollout.hit_max_tokens)
        self.assertFalse(rollout.stopped_on_answer)

    def test_student_rollout_enables_answer_stopping(self):
        sampled = GenerationResult((7,), "#### 7", False, False, stopped_on_answer=True)
        with patch("framework_opd.rollout._generate_text", return_value=sampled) as generate:
            rollout = generate_opd_rollout(
                None,
                None,
                FakeTokenizer(),
                "question",
                mode="vanilla",
                framework_max_new_tokens=4,
                solution_max_new_tokens=2,
                framework_temperature=0.0,
                solution_temperature=0.7,
            )

        self.assertTrue(generate.call_args.kwargs["stop_on_answer"])
        self.assertTrue(rollout.stopped_on_answer)

    def test_guided_rollout_routes_framework_and_solution_temperatures_separately(self):
        framework_result = FrameworkGenerationResult(
            ("Plan relation", "Check result"), 1, False, (), 2, 3, 0, False, True
        )
        sampled = GenerationResult((7,), "#### 7", False, False, stopped_on_answer=True)
        with (
            patch(
                "framework_opd.rollout.generate_framework_result",
                return_value=framework_result,
            ) as framework_generate,
            patch("framework_opd.rollout._generate_text", return_value=sampled) as solution_generate,
        ):
            generate_opd_rollout(
                None,
                None,
                FakeTokenizer(),
                "question",
                mode="guided",
                framework_max_new_tokens=4,
                solution_max_new_tokens=2,
                framework_temperature=0.0,
                solution_temperature=0.7,
            )

        self.assertEqual(framework_generate.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(solution_generate.call_args.args[4], 0.7)

    def test_framework_generation_retries_validation_failure(self):
        first = GenerationResult((3,), "1. bad\n2. bad\n</framework>", False, False)
        second = GenerationResult(
            (4,), "1. Plan relation\n2. Check result\n</framework>", False, False
        )
        with (
            patch("framework_opd.rollout._generate_text", side_effect=[first, second]) as generate,
            patch(
                "framework_opd.rollout.require_valid_framework",
                side_effect=[ValueError("leak"), ["Plan relation", "Check result"]],
            ),
        ):
            framework = generate_framework(
                None,
                FakeTokenizer(),
                "question",
                max_new_tokens=8,
                temperature=0.0,
                max_attempts=2,
            )

        self.assertEqual(framework, ["Plan relation", "Check result"])
        self.assertEqual(generate.call_count, 2)
        retry_prompt = generate.call_args_list[1].args[2]
        self.assertTrue(retry_prompt.endswith("<framework>\n"))
        self.assertLess(retry_prompt.index("previous attempt"), retry_prompt.rindex("<framework>"))

    def test_oracle_framework_generation_uses_reference_aware_prompt(self):
        generated = GenerationResult(
            (4,), "1. Plan relation\n2. Check result\n</framework>", False, False
        )
        with patch("framework_opd.rollout._generate_text", return_value=generated) as generate:
            result = generate_framework_result(
                None,
                FakeTokenizer(),
                "question",
                max_new_tokens=8,
                temperature=0.0,
                max_attempts=1,
                reference_answer="work\n#### 7",
            )

        self.assertFalse(result.used_fallback)
        prompt = generate.call_args.args[2]
        self.assertIn("Reference solution:\nwork\n#### 7", prompt)
        self.assertIn("Given a problem and its reference solution", prompt)

    def test_framework_generation_uses_recorded_fallback_after_all_attempts(self):
        invalid = GenerationResult((3,), "not numbered", False, False)
        with patch("framework_opd.rollout._generate_text", return_value=invalid) as generate:
            result = generate_framework_result(
                None,
                FakeTokenizer(),
                "question",
                max_new_tokens=8,
                temperature=0.0,
                max_attempts=2,
            )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.steps), 4)
        self.assertEqual(len(result.validation_errors), 2)
        self.assertEqual(result.validation_errors, ("framework_not_closed",) * 2)
        self.assertFalse(result.closed_tag)
        self.assertEqual(generate.call_count, 2)

    def test_framework_generation_accepts_complete_tag_before_length_limit_and_tracks_cost(self):
        truncated = GenerationResult(
            (3, 4),
            "1. Identify quantities\n2. Combine them\n</framework>",
            False,
            True,
            prompt_tokens=11,
        )
        with patch("framework_opd.rollout._generate_text", return_value=truncated):
            result = generate_framework_result(
                None,
                FakeTokenizer(),
                "question",
                max_new_tokens=8,
                temperature=0.0,
                max_attempts=2,
            )

        self.assertFalse(result.used_fallback)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(result.total_generated_tokens, 2)
        self.assertEqual(result.hit_max_attempts, 1)
        self.assertEqual(result.validation_errors, ())
        self.assertFalse(result.last_ended_with_eos)
        self.assertTrue(result.closed_tag)

    def test_framework_generation_rejects_missing_closing_tag_without_length_hit(self):
        unclosed = GenerationResult(
            (3,), "1. Identify quantities\n2. Combine them", True, False, prompt_tokens=7
        )
        with patch("framework_opd.rollout._generate_text", return_value=unclosed):
            result = generate_framework_result(
                None,
                FakeTokenizer(),
                "question",
                max_new_tokens=8,
                temperature=0.0,
                max_attempts=1,
            )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.validation_errors, ("framework_not_closed",))
        self.assertEqual(result.hit_max_attempts, 0)
        self.assertTrue(result.last_ended_with_eos)
        self.assertFalse(result.closed_tag)


if __name__ == "__main__":
    unittest.main()
