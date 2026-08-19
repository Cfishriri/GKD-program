import unittest

from framework_opd.prompts import format_student_prompt, format_vanilla_student_prompt
from framework_opd.rollout import validate_mode


class RolloutModeTest(unittest.TestCase):
    def test_vanilla_prompt_has_no_framework(self):
        prompt = format_vanilla_student_prompt("What is 2+3?")
        self.assertNotIn("<framework>", prompt)
        self.assertIn("What is 2+3?", prompt)
        self.assertTrue(prompt.endswith("<solution>\n"))

    def test_guided_prompt_contains_framework(self):
        prompt = format_student_prompt("What is 2+3?", ["Add the values"])
        self.assertIn("<framework>", prompt)
        self.assertIn("1. Add the values", prompt)

    def test_mode_validation(self):
        self.assertEqual(validate_mode("vanilla"), "vanilla")
        self.assertEqual(validate_mode("guided"), "guided")
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_mode("other")
