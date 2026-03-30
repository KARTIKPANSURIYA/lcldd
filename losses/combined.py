def combined_loss(
    L_ans,
    L_vf,
    L_lya,
    L_jac,
    lambda_vf=0.1,
    lambda_lya=0.1,
    lambda_jac=0.05,
    stage='G'   # curriculum stage A through G
):
    """
    L = L_ans + lambda_vf*L_vf + lambda_lya*L_lya + lambda_jac*L_jac
    Curriculum: introduce terms progressively
    """
    loss = L_ans

    if stage in ['C','D','E','F','G']:
        loss += lambda_vf * L_vf

    if stage in ['E','F','G']:
        loss += lambda_lya * L_lya

    if stage in ['F','G']:
        loss += lambda_jac * L_jac

    return loss
