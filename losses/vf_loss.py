"""
Vector field distillation loss.

Matches per-step velocity between student and teacher trajectories:

    v_t = h_{t+1} - h_t
    L_vf = (1 / T) * sum_{t=0..T-1}  mean_batch[ || v_t^S - v_t^T ||^2 ]

Notes
-----
- Teacher trajectory comes from the precomputed cache (last N hidden
  layers of the 32B teacher, mean-pooled over seq and projected to the
  student's hidden dim).  Student trajectory comes from T_MAX recursive
  applications of the thinking block.  The two dynamics are conceptually
  different (across transformer layers vs across iterative thinking
  steps), but the LCLDD hypothesis is that they should align.
- Uses `(v_s - v_t).pow(2).sum(-1)` rather than `torch.norm(...).pow(2)`
  to skip an unnecessary sqrt-then-square.
- Returns a grad-safe zero on the degenerate T<=0 case.
"""
import torch


def vector_field_loss(student_trajectory, teacher_trajectory):
    """
    Args:
        student_trajectory : list of tensors [(B, d), ...]
        teacher_trajectory : list of tensors [(B, d), ...] (no grad OK)

    Returns:
        scalar tensor on student_trajectory[0]'s device
    """
    T = min(len(student_trajectory), len(teacher_trajectory)) - 1
    if T <= 0:
        return student_trajectory[0].new_zeros(())

    total = student_trajectory[0].new_zeros(())
    for t in range(T):
        v_s = student_trajectory[t + 1] - student_trajectory[t]
        v_t = teacher_trajectory[t + 1] - teacher_trajectory[t]
        total = total + (v_s - v_t).pow(2).sum(dim=-1).mean()
    return total / T