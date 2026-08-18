import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute TRL-compatible generalized JSD on completion positions."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if completion_mask.shape != student_logits.shape[:2]:
        raise ValueError("completion_mask must match logits batch and sequence dimensions")

    token_count = completion_mask.sum()
    if token_count.item() == 0:
        raise ValueError("at least one completion token is required")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)

    if beta == 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        weight = student_log_probs.new_tensor(beta)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [student_log_probs + torch.log1p(-weight), teacher_log_probs + torch.log(weight)]
            ),
            dim=0,
        )
        teacher_kl = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        student_kl = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = weight * teacher_kl + (1.0 - weight) * student_kl

    per_token = per_vocab.sum(dim=-1)
    loss = (per_token * completion_mask.to(per_token.dtype)).sum() / token_count
    return loss, {
        "loss": loss.detach(),
        "num_tokens": token_count.detach(),
        "mean_student_entropy": (
            -(student_log_probs.exp() * student_log_probs).sum(dim=-1) * completion_mask
        ).sum().detach()
        / token_count,
    }
