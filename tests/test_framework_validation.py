import unittest

from framework_opd.framework_validation import (
    FrameworkValidationError,
    audit_framework_records,
    normalise_final_answer,
    require_valid_framework,
    validate_framework,
)
from framework_opd.prompts import (
    STUDENT_SYSTEM_PROMPT,
    VANILLA_STUDENT_SYSTEM_PROMPT,
    extract_framework,
    format_framework_prompt,
    has_complete_framework,
)


class FrameworkValidationTest(unittest.TestCase):
    def test_accepts_abstract_two_to_six_step_plan(self):
        steps = ["Identify the relevant quantities", "Combine them and check the result"]
        result = validate_framework(steps, "work\n#### 42")
        self.assertTrue(result.valid)
        self.assertEqual(result.reasons, ())

    def test_rejects_step_counts_outside_range_without_parser_truncation(self):
        too_many_text = "<framework>\n" + "\n".join(f"{index}. abstract step" for index in range(1, 8))
        steps = extract_framework(too_many_text)
        self.assertEqual(len(steps), 7)
        self.assertIn("step_count", validate_framework(steps).reasons)
        self.assertIn("step_count", validate_framework(["only one step"]).reasons)

    def test_rejects_answer_marker_and_numeric_literals(self):
        result = validate_framework(["Identify the values", "Report #### 9"], "#### 9")
        self.assertIn("answer_marker", result.reasons)
        self.assertIn("numeric_literal", result.reasons)
        self.assertIn("final_answer_leak", result.reasons)

    def test_rejects_numbers_spelled_out_in_english_or_chinese(self):
        english = validate_framework(
            ["Identify the relevant quantity", "Return forty-two as the result"]
        )
        chinese = validate_framework(["识别相关数量", "结果为四十二"])
        self.assertIn("number_word", english.reasons)
        self.assertIn("number_word", chinese.reasons)

    def test_rejects_quantifier_synonyms_ordinals_and_derived_number_words(self):
        leaked_words = (
            "single",
            "both",
            "pair",
            "couple",
            "dozen",
            "score",
            "gross",
            "half",
            "quarter",
            "nought",
            "nil",
            "duo",
            "trio",
            "quartet",
            "quintet",
            "sextet",
            "septet",
            "octet",
            "nonet",
            "first",
            "twenty-first",
            "twofold",
            "three-fold",
        )
        for leaked_word in leaked_words:
            with self.subTest(leaked_word=leaked_word):
                result = validate_framework(
                    ["Identify the relevant quantities", f"Use the {leaked_word} relation"]
                )
                self.assertIn("number_word", result.reasons)

    def test_rejects_unicode_numeric_and_uppercase_ascii_roman_numerals(self):
        for numeric in ("３", "²", "Ⅳ", "I", "IV", "IIII"):
            with self.subTest(numeric=numeric):
                result = validate_framework(
                    ["Identify the relevant quantities", f"Use group {numeric} in the model"]
                )
                self.assertIn("numeric_literal", result.reasons)

    def test_allows_unevaluated_relational_operations(self):
        result = validate_framework(
            [
                "Halve or double the relevant amount as required by the relationship",
                "Check whether a quantity is triple or twice another",
            ]
        )
        self.assertTrue(result.valid, result.reasons)

    def test_rejects_evaluated_equation_even_when_numbers_are_allowed(self):
        result = validate_framework(
            ["Identify the inputs", "Calculate 12 * 52 = 624 pages"],
            "#### 100",
            forbid_numbers=False,
        )
        self.assertIn("evaluated_equation", result.reasons)
        self.assertNotIn("numeric_literal", result.reasons)

    def test_normalized_final_answer_matching_handles_commas_and_decimal_zeros(self):
        self.assertEqual(normalise_final_answer("reasoning\n#### 1,200.00"), "1200")
        result = validate_framework(
            ["Identify the inputs", "The resulting total is 1200"],
            "reasoning\n#### 1,200.00",
            forbid_numbers=False,
        )
        self.assertIn("final_answer_leak", result.reasons)

    def test_require_valid_framework_raises_typed_error(self):
        with self.assertRaises(FrameworkValidationError) as context:
            require_valid_framework(["Use 3 items", "Return the result"])
        self.assertIn("numeric_literal", context.exception.reasons)

    def test_rejects_non_string_or_empty_steps(self):
        result = validate_framework(["Identify the quantities", None])
        self.assertIn("non_string_step", result.reasons)
        self.assertIn("empty_step", result.reasons)

    def test_audit_contains_required_counts_and_leakage_rate(self):
        records = [
            {
                "question": "question",
                "answer": "#### 5",
                "framework": ["Identify the quantities", "Combine them"],
            },
            {
                "question": "question",
                "answer": "#### 5",
                "framework": ["Identify the quantities", "The result is 5"],
            },
        ]
        audit = audit_framework_records(records)
        self.assertEqual(audit["valid"], 1)
        self.assertEqual(audit["invalid"], 1)
        self.assertEqual(audit["reasons"]["final_answer_leak"], 1)
        self.assertEqual(audit["leakage_rate"], 0.5)

    def test_retry_prompt_repeats_purity_constraint(self):
        prompt = format_framework_prompt("question", "#### 5", ["numeric_literal"])
        self.assertIn("previous attempt was rejected", prompt)
        self.assertIn("numeric_literal", prompt)
        self.assertTrue(prompt.endswith("<framework>\n"))

    def test_strict_framework_extraction_requires_closing_tag(self):
        truncated = "<framework>\n1. Identify quantities\n2. Combine them"
        self.assertFalse(has_complete_framework(truncated))
        with self.assertRaisesRegex(ValueError, "framework_not_closed"):
            extract_framework(truncated, require_closed=True)
        closed = truncated + "\n</framework>"
        self.assertTrue(has_complete_framework(closed))
        self.assertEqual(
            extract_framework(closed, require_closed=True),
            ["Identify quantities", "Combine them"],
        )

    def test_student_prompts_share_concise_non_repetition_instruction(self):
        shared = "Give concise calculations and reasoning without restating or repeating the problem."
        self.assertIn(shared, STUDENT_SYSTEM_PROMPT)
        self.assertIn(shared, VANILLA_STUDENT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
