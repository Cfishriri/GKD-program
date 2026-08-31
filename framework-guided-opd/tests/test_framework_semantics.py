import unittest
from unittest.mock import patch

from framework_opd.framework_semantics import verify_framework_semantics
from framework_opd.rollout import GenerationResult


class FrameworkSemanticsTest(unittest.TestCase):
    def test_accepts_only_strict_reference_matching_execution(self):
        generation = GenerationResult(
            (4, 5),
            "reasoning\n#### 20",
            False,
            False,
            prompt_tokens=17,
            stopped_on_answer=True,
        )
        with patch(
            "framework_opd.framework_semantics._generate_text", return_value=generation
        ) as generate:
            result = verify_framework_semantics(
                object(),
                object(),
                "question",
                ["Identify the quantities", "Combine them"],
                "reference work\n#### 20",
                max_new_tokens=64,
            )

        self.assertTrue(result.correct)
        self.assertEqual(result.predicted_answer, "20")
        self.assertEqual(result.reference_answer, "20")
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(result.prompt_tokens, 17)
        self.assertTrue(result.stopped_on_answer)
        self.assertTrue(generate.call_args.kwargs["stop_on_answer"])
        self.assertEqual(generate.call_args.args[4], 0.0)

    def test_rejects_wrong_or_non_strict_execution(self):
        for text, predicted in (("#### 60", "60"), ("the result is 20", None)):
            with self.subTest(text=text), patch(
                "framework_opd.framework_semantics._generate_text",
                return_value=GenerationResult((7,), text, False, False),
            ):
                result = verify_framework_semantics(
                    object(), object(), "question", ["Plan", "Check"], "#### 20", 64
                )
                self.assertFalse(result.correct)
                self.assertEqual(result.predicted_answer, predicted)

    def test_semantic_generation_budget_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 2048"):
            verify_framework_semantics(
                object(), object(), "question", ["Plan", "Check"], "#### 20", 2049
            )


if __name__ == "__main__":
    unittest.main()
