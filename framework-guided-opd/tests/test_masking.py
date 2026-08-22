import unittest

import torch

from framework_opd.masking import causal_completion_mask


class MaskingTest(unittest.TestCase):
    def test_causal_mask_marks_logits_predicting_completion_tokens(self):
        labels = torch.tensor([[-100, -100, 11, 12, -100]])
        self.assertEqual(causal_completion_mask(labels).tolist(), [[0.0, 1.0, 1.0, 0.0]])

    def test_causal_mask_handles_batch(self):
        labels = torch.tensor([[-100, 4, 5], [-100, -100, 7]])
        self.assertEqual(causal_completion_mask(labels).tolist(), [[1.0, 1.0], [0.0, 1.0]])
