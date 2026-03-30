import torch

def jacobian_loss(h_T_student, h_T_teacher, x_embed):
    B, d = h_T_student.shape
    
    # Compute Jacobian for student
    jac_rows_student = []
    for i in range(0, d, 64):
        v = torch.zeros_like(h_T_student)
        v[:, i:min(i+64, d)] = 1.0
        grad = torch.autograd.grad(
            h_T_student, x_embed,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        if grad is None:
            grad = torch.zeros_like(x_embed)
        jac_rows_student.append(grad)
    J_student = torch.stack(jac_rows_student, dim=1)
    
    # Compute Jacobian for teacher
    jac_rows_teacher = []
    for i in range(0, d, 64):
        v = torch.zeros_like(h_T_teacher)
        v[:, i:min(i+64, d)] = 1.0
        grad = torch.autograd.grad(
            h_T_teacher, x_embed,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        if grad is None:
            grad = torch.zeros_like(x_embed)
        jac_rows_teacher.append(grad)
    J_teacher = torch.stack(jac_rows_teacher, dim=1)
    
    return torch.norm(J_student - J_teacher, p='fro', dim=(-2,-1)).mean()