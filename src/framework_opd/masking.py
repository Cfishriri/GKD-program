import torch


def causal_completion_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a mask aligned with causal logits at positions ``[:-1]``."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    return labels[:, 1:].ne(-100).to(dtype=torch.float32)

