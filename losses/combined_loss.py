import torch

def combined_loss(L_ans, L_vf, L_lya, L_jac, lambda_vf=0.1, lambda_lya=0.1, lambda_jac=0.05, stage='A'):
    # Base loss
    loss = L_ans
    
    if stage in ['D']:
        loss = loss + lambda_vf * L_vf
    elif stage in ['E']:
        loss = loss + lambda_vf * L_vf + lambda_lya * L_lya
    elif stage in ['F', 'G']:
        loss = loss + lambda_vf * L_vf + lambda_lya * L_lya + lambda_jac * L_jac
        
    return loss
