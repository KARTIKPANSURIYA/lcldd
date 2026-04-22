"""
Combined LCLDD loss with 7-stage curriculum.

    L = L_ans + lambda_vf * L_vf + lambda_lya * L_lya + lambda_jac * L_jac

Curriculum
----------
    A, B, C  ->  L_ans only            (warmup on cross-entropy)
    D        ->  + L_vf                (vector-field distillation)
    E        ->  + L_lya               (Lyapunov stability)
    F, G     ->  + L_jac               (Jacobian manifold alignment)

Default lambda values
---------------------
These defaults assume the auxiliary losses are NOT pre-normalized by
hidden_dim in the caller.  They are the raw per-batch means returned by
lyapunov_loss / vector_field_loss / jacobian_loss.

Empirically:
    L_ans ~ O(1)      (cross-entropy)
    L_lya ~ O(1)-O(10)   with the contractive thinking block
    L_vf  ~ O(10)-O(100) since teacher-projected velocities have larger norm
    L_jac ~ O(1)-O(10)   (Hutchinson estimator on unit directions)

The defaults below put all three auxiliaries at roughly comparable
contribution to the total.  Retune lambda_vf if your teacher projection
head produces unusually large-magnitude trajectories.

IMPORTANT — do NOT divide the auxiliary losses by hidden_dim in the
training loop (e.g., `L_lya / HIDDEN_DIM`). Doing so shrinks them by a
factor of 3584 and silently zeroes out their effective contribution.
"""


def combined_loss(
    L_ans,
    L_vf,
    L_lya,
    L_jac,
    lambda_vf=0.01,
    lambda_lya=1.0,
    lambda_jac=0.001,
    stage='E',
):
    """
    Args:
        L_ans  : scalar cross-entropy loss on the target answer token(s)
        L_vf   : scalar vector-field loss (zeroed out by caller before stage D)
        L_lya  : scalar Lyapunov loss    (zeroed out by caller before stage E)
        L_jac  : scalar Jacobian loss    (zeroed out by caller before stage F)
        lambda_vf, lambda_lya, lambda_jac : mixing coefficients
        stage  : one of {'A','B','C','D','E','F','G'}

    Returns:
        scalar total loss
    """
    if stage not in ('A', 'B', 'C', 'D', 'E', 'F', 'G'):
        raise ValueError(f"Unknown curriculum stage: {stage!r}")

    loss = L_ans

    if stage in ('D', 'E', 'F', 'G'):
        loss = loss + lambda_vf * L_vf

    if stage in ('E', 'F', 'G'):
        loss = loss + lambda_lya * L_lya

    if stage in ('F', 'G'):
        loss = loss + lambda_jac * L_jac

    return loss