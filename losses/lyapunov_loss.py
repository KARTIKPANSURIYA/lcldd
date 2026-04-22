"""
Lyapunov stability loss.

Hinge penalty for ascent of the Lyapunov energy V(h_t) along a trajectory:

    L_lya = (1 / T) * sum_{t=0..T-1}  mean_batch[ relu( V(h_{t+1}) - V(h_t) + eps ) ]

Zero iff V descends (by at least eps) at every consecutive step.  This is
the operational form of Proposition 1: train the thinking block so that
L_lya -> 0, i.e., energy is strictly monotonically non-increasing along
every iterated trajectory.

Notes
-----
- `epsilon` is a small positive margin; increasing it enforces a strictly
  larger descent per step.  With V on the order of 1 for a well-trained
  block, eps=1e-3 is effectively zero-margin (non-increase). If V is large
  in your regime, consider raising eps to ~1% of typical V.
- The function is written to preserve the autograd graph (so gradients
  flow back into the thinking block parameters via V) and to return a
  grad-safe zero on the degenerate T=0 case.
"""
import torch
import torch.nn.functional as F


def lyapunov_loss(thinking_block, trajectory, phi_x, epsilon=1e-3):
    """
    Args:
        thinking_block : module implementing .lyapunov_energy(h, phi_x)
        trajectory     : list of tensors [(B, d), ...] of length T_MAX + 1
        phi_x          : (B, d) anchor
        epsilon        : float, descent margin

    Returns:
        scalar tensor (on the same device/dtype as trajectory[0])
    """
    T = len(trajectory) - 1
    if T <= 0:
        return trajectory[0].new_zeros(())

    total = trajectory[0].new_zeros(())
    for t in range(T):
        V_t  = thinking_block.lyapunov_energy(trajectory[t],     phi_x)
        V_t1 = thinking_block.lyapunov_energy(trajectory[t + 1], phi_x)
        total = total + F.relu(V_t1 - V_t + epsilon).mean()
    return total / T