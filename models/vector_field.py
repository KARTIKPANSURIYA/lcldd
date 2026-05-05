"""
Vector-field utilities for LCLDD.

This module contains reusable helpers for computing trajectory velocities and
vector-field alignment losses. The main training script currently imports the
loss from ``losses/vf_loss.py`` for backward compatibility, but these utilities
are kept here as model-side helpers for experiments, ablations, and future
extensions.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def compute_velocities(trajectory: Iterable[Tensor]) -> List[Tensor]:
    """Return first-order velocity vectors for a latent trajectory.

    Args:
        trajectory: Iterable of tensors with shape ``(batch, hidden_dim)``.

    Returns:
        List of tensors ``trajectory[t + 1] - trajectory[t]``.
    """
    states = list(trajectory)
    if len(states) < 2:
        return []
    return [states[t + 1] - states[t] for t in range(len(states) - 1)]


def match_trajectory_length(student_trajectory: Iterable[Tensor], teacher_trajectory: Iterable[Tensor]):
    """Trim student and teacher trajectories to the same comparable length."""
    student_states = list(student_trajectory)
    teacher_states = list(teacher_trajectory)
    length = min(len(student_states), len(teacher_states))
    return student_states[:length], teacher_states[:length]


class VectorFieldAligner(nn.Module):
    """Vector-field alignment objective for teacher-student latent dynamics.

    The objective compares the motion between consecutive latent states rather
    than only comparing final hidden states. This supports the LCLDD hypothesis
    that a student should imitate the teacher's direction of reasoning in latent
    space.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")
        self.reduction = reduction

    def forward(self, student_trajectory: Iterable[Tensor], teacher_trajectory: Iterable[Tensor]) -> Tensor:
        student_states, teacher_states = match_trajectory_length(student_trajectory, teacher_trajectory)
        student_velocities = compute_velocities(student_states)
        teacher_velocities = compute_velocities(teacher_states)

        if not student_velocities:
            if student_states:
                return student_states[0].new_zeros(())
            raise ValueError("student_trajectory must contain at least one tensor")

        losses = []
        for v_s, v_t in zip(student_velocities, teacher_velocities):
            v_t = v_t.to(device=v_s.device, dtype=v_s.dtype)
            losses.append(F.mse_loss(v_s, v_t, reduction="none").sum(dim=-1))

        stacked = torch.stack(losses, dim=0)  # (steps, batch)

        if self.reduction == "mean":
            return stacked.mean()
        if self.reduction == "sum":
            return stacked.sum()
        return stacked


def vector_field_alignment_loss(student_trajectory: Iterable[Tensor], teacher_trajectory: Iterable[Tensor]) -> Tensor:
    """Functional wrapper for mean vector-field alignment loss."""
    return VectorFieldAligner(reduction="mean")(student_trajectory, teacher_trajectory)
