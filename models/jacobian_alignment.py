"""
Jacobian alignment utilities for LCLDD.

Jacobian alignment was part of the broader LCLDD design as an optional future
extension. It is not required for the main Stage A/D/E experiments, but this file
provides a working, documented implementation for experiments that need to align
student and teacher input sensitivity.

The implementation uses a lightweight Hutchinson-style directional derivative
estimate instead of forming the full Jacobian matrix, which would be too costly
for large language-model hidden states.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _rademacher_like(x: Tensor) -> Tensor:
    """Sample Rademacher noise with values in {-1, +1}."""
    return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def directional_jvp(
    fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    direction: Optional[Tensor] = None,
    create_graph: bool = True,
) -> Tensor:
    """Estimate a Jacobian-vector product for ``fn`` at ``inputs``.

    Args:
        fn: Function mapping a tensor to a tensor with the same batch dimension.
        inputs: Input tensor that should receive gradients.
        direction: Optional probe vector. If omitted, Rademacher noise is used.
        create_graph: Whether to keep the graph for higher-order optimization.

    Returns:
        Directional derivative with the same shape as ``fn(inputs)``.
    """
    x = inputs.detach().requires_grad_(True)
    v = direction if direction is not None else _rademacher_like(x)
    y = fn(x)

    # Convert vector output to scalar directional projection, then differentiate.
    projection = (y * v[..., : y.shape[-1]]).sum()
    grad = torch.autograd.grad(
        projection,
        x,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return grad


class JacobianAlignmentLoss(nn.Module):
    """Hutchinson-style Jacobian alignment loss.

    This loss compares student and teacher directional sensitivities without
    materializing full Jacobians. It is intended for optional Stage F/G research
    runs, not for the default Stage E pipeline.
    """

    def __init__(self, probes: int = 1, reduction: str = "mean") -> None:
        super().__init__()
        if probes < 1:
            raise ValueError("probes must be >= 1")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.probes = probes
        self.reduction = reduction

    def forward(
        self,
        student_fn: Callable[[Tensor], Tensor],
        teacher_fn: Callable[[Tensor], Tensor],
        inputs: Tensor,
    ) -> Tensor:
        losses = []
        for _ in range(self.probes):
            direction = _rademacher_like(inputs)
            j_student = directional_jvp(student_fn, inputs, direction, create_graph=True)
            with torch.no_grad():
                # Teacher sensitivity is treated as a fixed target.
                j_teacher = directional_jvp(teacher_fn, inputs, direction, create_graph=False)
            losses.append(F.mse_loss(j_student, j_teacher.to(j_student.device), reduction="mean"))

        stacked = torch.stack(losses)
        if self.reduction == "sum":
            return stacked.sum()
        return stacked.mean()


def jacobian_alignment_loss(
    student_fn: Callable[[Tensor], Tensor],
    teacher_fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    probes: int = 1,
) -> Tensor:
    """Functional wrapper for Jacobian alignment."""
    return JacobianAlignmentLoss(probes=probes)(student_fn, teacher_fn, inputs)
