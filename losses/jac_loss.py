"""
Jacobian manifold alignment — Hutchinson-style stochastic estimator.

Motivation
----------
A full Jacobian J = d(h_T) / d(phi_x) for d=3584 has 12.8M entries per
sample — impractical to form or match directly.  Instead we match
vector-Jacobian products (VJPs) along shared random directions v:

    L_jac  =  E_v [ || J_S^T v  -  (J_T^T v)_cached ||^2 ]

PyTorch gives us VJPs cheaply via scalar-backward:

    scalar = sum( h_T . v )
    J^T @ v  =  d(scalar) / d(phi_x)      # one backward pass

Matching J_S^T v against J_T^T v is equivalent to matching J_S @ v
against J_T @ v in expectation over random v (up to a constant), so this
is a valid Hutchinson estimator of || J_S - J_T ||_F^2.

The teacher VJP must be precomputed at teacher-cache time using the SAME
set of random directions {v_k}, and must live in the student's hidden
dim (project via the same proj: R^d_t -> R^d_s used for the hidden
states).  See `precompute_teacher_jvps()` in the training notebook.

Caller responsibilities
-----------------------
- `phi_x` must have `requires_grad=True` when the thinking-block
  iteration begins (otherwise J^S has no graph to phi_x).
- `h_T_student` must retain its autograd graph — do NOT .detach() it.
- `create_graph=True` here makes Stage F roughly 2x slower than Stage E,
  because gradient of the VJP must flow back through the thinking block
  parameters when we call backward on the total loss.

Why not the original version
----------------------------
The old `jac_loss.py` called `torch.autograd.grad(h_T_teacher, x_embed,
..., allow_unused=True)` on a teacher tensor loaded from a cache saved
under `torch.no_grad()`.  Autograd returned `None` and the code silently
replaced it with `torch.zeros_like(x_embed)`.  That makes the teacher
Jacobian identically zero, so `L_jac = ||J_S - 0||^2` — which trains the
student to be input-invariant.  This file fixes that by requiring
precomputed, non-zero teacher VJPs.
"""
import torch


def jacobian_loss(h_T_student, phi_x, teacher_vjp, v):
    """
    One-direction Hutchinson term.  Average over K directions in the caller.

    Args:
        h_T_student : (B, d)   student thinking-block final state, WITH grad graph
        phi_x       : (B, d)   student anchor, requires_grad=True
        teacher_vjp : (B, d)   cached J_T^T @ v in student hidden dim, no grad
        v           : (d,) or (B, d)   the random direction used at cache time

    Returns:
        scalar loss
    """
    if h_T_student.shape != phi_x.shape:
        raise ValueError(
            f"Shape mismatch: h_T_student {tuple(h_T_student.shape)} vs "
            f"phi_x {tuple(phi_x.shape)}"
        )
    if not phi_x.requires_grad:
        raise RuntimeError(
            "phi_x must have requires_grad=True before the thinking-block "
            "iteration starts. See the Stage F training cell for the fix."
        )

    # Broadcast v from (d,) to (B, d) if needed.
    if v.dim() == 1:
        v = v.unsqueeze(0).expand_as(h_T_student)

    # Scalar whose gradient w.r.t. phi_x equals J_S^T @ v (per batch row).
    scalar = (h_T_student * v).sum()

    jvp_student, = torch.autograd.grad(
        scalar, phi_x,
        retain_graph=True,
        create_graph=True,   # needed so L_jac.backward() updates tb params
    )

    # teacher_vjp is detached by construction (loaded from cache).
    return (jvp_student - teacher_vjp).pow(2).sum(dim=-1).mean()


def jacobian_loss_multi(h_T_student, phi_x, teacher_vjps, vs):
    """
    Convenience wrapper: average jacobian_loss over K cached directions.

    Args:
        h_T_student  : (B, d)
        phi_x        : (B, d), requires_grad=True
        teacher_vjps : (K, B, d) stack of cached J_T^T v_k
        vs           : (K, d)    stack of random directions

    Returns:
        scalar loss averaged over K
    """
    K = teacher_vjps.shape[0]
    total = h_T_student.new_zeros(())
    for k in range(K):
        total = total + jacobian_loss(h_T_student, phi_x, teacher_vjps[k], vs[k])
    return total / K