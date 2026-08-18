import unittest
import torch

from framework_opd.loss import generalized_jsd_loss


class LossTest(unittest.TestCase):
    def test_identical_distributions_have_zero_loss(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
        loss, metrics = generalized_jsd_loss(logits, logits.detach(), torch.ones(1, 1), beta=0.5)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertEqual(metrics["num_tokens"].item(), 1)

    def test_mask_excludes_prompt_positions(self):
        student = torch.tensor([[[20.0, -20.0], [1.0, 2.0]]], requires_grad=True)
        teacher = torch.tensor([[[-20.0, 20.0], [2.0, 1.0]]])
        masked_loss, _ = generalized_jsd_loss(student, teacher, torch.tensor([[0.0, 1.0]]), beta=0.5)
        expected_loss, _ = generalized_jsd_loss(student[:, 1:], teacher[:, 1:], torch.ones(1, 1), beta=0.5)
        self.assertAlmostEqual(masked_loss.item(), expected_loss.item(), places=6)

    def test_teacher_is_always_detached(self):
        student = torch.randn(1, 2, 3, requires_grad=True)
        teacher = torch.randn(1, 2, 3, requires_grad=True)
        loss, _ = generalized_jsd_loss(student, teacher, torch.tensor([[0.0, 1.0]]), beta=0.5)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_loss_requires_a_completion_token(self):
        with self.assertRaisesRegex(ValueError, "completion token"):
            generalized_jsd_loss(torch.randn(1, 2, 3), torch.randn(1, 2, 3), torch.zeros(1, 2), beta=0.5)
