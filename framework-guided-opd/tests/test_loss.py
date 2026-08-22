import unittest
import torch

from framework_opd.loss import generalized_jsd_loss


class LossTest(unittest.TestCase):
    def test_identical_distributions_have_zero_loss(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
        loss, metrics = generalized_jsd_loss(logits, logits.detach(), torch.ones(1, 1), beta=0.5)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertEqual(metrics["num_tokens"].item(), 1)
        self.assertAlmostEqual(metrics["divergence_sum"].item(), 0.0, places=6)

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

    def test_sum_reduction_exposes_token_global_accumulation_values(self):
        student = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]], requires_grad=True
        )
        teacher = torch.tensor([[[0.0, 2.0], [2.0, 0.0], [-1.0, 1.0]]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])

        summed, metrics = generalized_jsd_loss(
            student, teacher, mask, beta=1.0, reduction="sum"
        )
        mean, _ = generalized_jsd_loss(student, teacher, mask, beta=1.0)

        self.assertEqual(metrics["num_tokens"].item(), 2)
        self.assertAlmostEqual(summed.item(), metrics["divergence_sum"].item(), places=6)
        self.assertAlmostEqual(summed.item() / 2, mean.item(), places=6)

    def test_reduction_must_be_supported(self):
        with self.assertRaisesRegex(ValueError, "reduction"):
            generalized_jsd_loss(
                torch.randn(1, 1, 3),
                torch.randn(1, 1, 3),
                torch.ones(1, 1),
                reduction="batchmean",
            )

    def test_multiple_rollouts_are_normalized_by_total_tokens(self):
        student_short = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
        teacher_short = torch.tensor([[[0.0, 2.0]]])
        student_long = torch.tensor([[[0.0, 2.0], [1.0, -1.0]]], requires_grad=True)
        teacher_long = torch.tensor([[[2.0, 0.0], [-1.0, 1.0]]])

        short_sum, short_metrics = generalized_jsd_loss(
            student_short, teacher_short, torch.ones(1, 1), beta=1.0, reduction="sum"
        )
        long_sum, long_metrics = generalized_jsd_loss(
            student_long, teacher_long, torch.ones(1, 2), beta=1.0, reduction="sum"
        )
        combined_mean, _ = generalized_jsd_loss(
            torch.cat([student_short, student_long], dim=1),
            torch.cat([teacher_short, teacher_long], dim=1),
            torch.ones(1, 3),
            beta=1.0,
        )

        total_tokens = short_metrics["num_tokens"] + long_metrics["num_tokens"]
        window_mean = (short_sum + long_sum) / total_tokens
        self.assertAlmostEqual(window_mean.item(), combined_mean.item(), places=6)
