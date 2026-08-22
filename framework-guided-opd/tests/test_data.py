import unittest

from framework_opd.data import validate_record
from framework_opd.prompts import (
    extract_framework,
    format_framework_prompt,
    format_student_prompt,
)


class DataTest(unittest.TestCase):
    def test_validate_gsm8k_record(self):
        self.assertEqual(validate_record({"question": "2+3?", "answer": "#### 5"}), {
            "question": "2+3?", "answer": "#### 5"
        })

    def test_validate_record_rejects_missing_question(self):
        with self.assertRaisesRegex(ValueError, "question"):
            validate_record({"answer": "5"})

    def test_framework_training_prompt_contains_reference_answer(self):
        prompt = format_framework_prompt("2+3?", "#### 5")
        self.assertIn("2+3?", prompt)
        self.assertIn("#### 5", prompt)
        self.assertTrue(prompt.endswith("<framework>\n"))

    def test_student_prompt_does_not_contain_reference_answer(self):
        prompt = format_student_prompt("2+3?", ["理解题意", "完成计算"])
        self.assertNotIn("#### 5", prompt)
        self.assertIn("1. 理解题意", prompt)
        self.assertTrue(prompt.endswith("<solution>\n"))

    def test_extract_framework_accepts_numbered_steps(self):
        text = "<framework>\n1. 理解题意\n2、建立算式\n3) 检查答案\n</framework>"
        self.assertEqual(extract_framework(text), ["理解题意", "建立算式", "检查答案"])

    def test_extract_framework_rejects_empty_output(self):
        with self.assertRaisesRegex(ValueError, "numbered steps"):
            extract_framework("I cannot solve it")
