import unittest

from framework_opd.evaluation import extract_final_answer, score_prediction, summarize


class EvaluationTest(unittest.TestCase):
    def test_extracts_marked_answer_and_normalizes_currency(self):
        self.assertEqual(extract_final_answer("Result: #### $1,234.50"), "1234.5")

    def test_falls_back_to_last_number(self):
        self.assertEqual(extract_final_answer("First 4, finally 9."), "9")

    def test_exact_numeric_match(self):
        result = score_prediction("#### 5.0", "work\n#### 5")
        self.assertTrue(result["correct"])
        self.assertTrue(result["has_answer_marker"])

    def test_summary_reports_accuracy_and_format_rate(self):
        summary = summarize([
            {"correct": True, "has_answer_marker": True},
            {"correct": False, "has_answer_marker": False},
        ])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["answer_format_rate"], 0.5)
        self.assertLess(summary["accuracy_ci95_low"], summary["accuracy"])
        self.assertGreater(summary["accuracy_ci95_high"], summary["accuracy"])
