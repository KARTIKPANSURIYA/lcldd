import torch

def vector_field_loss(student_trajectory, teacher_trajectory):
    total_loss = 0.0
    T = min(len(student_trajectory), len(teacher_trajectory)) - 1
    
    if T <= 0:
        return torch.tensor(0.0)
        
    for t in range(T):
        v_student = student_trajectory[t+1] - student_trajectory[t]
        v_teacher = teacher_trajectory[t+1] - teacher_trajectory[t]
        # Mean over batch for this step
        step_loss = torch.norm(v_student - v_teacher, dim=-1).pow(2).mean()
        total_loss += step_loss
        
    return total_loss / T