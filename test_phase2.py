import torch
from models.thinking_block import LyapunovThinkingBlock
from models.halting import PhaseSpaceHalting
from losses.lyapunov_loss import lyapunov_loss
from losses.vf_loss import vector_field_loss
from losses.jac_loss import jacobian_loss
from losses.combined_loss import combined_loss

def main():
    # Phase 2
    block = LyapunovThinkingBlock(hidden_dim=256)
    phi_x = torch.randn(4, 256)
    h_t = phi_x.clone()
    
    trajectory = [h_t]
    for _ in range(5):
        h_t = block(h_t, phi_x)
        trajectory.append(h_t)
        
    loss = lyapunov_loss(block, trajectory, phi_x)
    print(f"Lyapunov Loss: {loss.item():.4f}")
    if isinstance(loss, torch.Tensor) and not torch.isnan(loss).any():
        print("Phase 2 complete")
        
    # Phase 3
    T_plus_1 = len(trajectory)
    teacher_trajectory = [torch.randn(4, 256) for _ in range(T_plus_1)]
    
    vf_loss = vector_field_loss(trajectory, teacher_trajectory)
    print(f"VF Loss: {vf_loss.item():.4f}")
    if isinstance(vf_loss, torch.Tensor) and not torch.isnan(vf_loss).any():
        print("Phase 3 complete")

    # Phase 4
    x_embed = torch.randn(4, 10, 256, requires_grad=True)
    h_T_student = torch.randn(4, 256, requires_grad=True)
    h_T_teacher = torch.randn(4, 256, requires_grad=True)
    
    h_T_student = h_T_student + x_embed.mean(dim=1)
    h_T_teacher = h_T_teacher + x_embed.mean(dim=1) * 0.5
    
    jac_loss = jacobian_loss(h_T_student, h_T_teacher, x_embed)
    print(f"Jacobian Loss: {jac_loss.item():.4f}")
    if isinstance(jac_loss, torch.Tensor) and not torch.isnan(jac_loss).any():
        print("Phase 4 complete")
        
    # Phase 5
    halter = PhaseSpaceHalting()
    
    h_t_halt = torch.randn(4, 256)
    h_prev_halt = torch.randn(4, 256)
    energy_t = torch.rand(4)
    logits = torch.randn(4, 100) # 100 vocab classes
    
    halt, delta_t, kappa_t, eta_t = halter(h_t_halt, h_prev_halt, energy_t, logits)
    print("Halt mask:", halt)
    print("Delta:", delta_t.mean().item())
    print("Confidence:", kappa_t.mean().item())
    print("Entropy:", eta_t.mean().item())
    
    if isinstance(halt, torch.Tensor) and halt.dtype == torch.bool:
        print("Phase 5 complete")
        
    # Phase 6
    L_ans = torch.tensor(2.5, requires_grad=True)
    L_vf  = torch.tensor(772.0)
    L_lya = torch.tensor(1069.0)
    L_jac = torch.tensor(1.26)

    stages = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    all_valid = True
    for stage in stages:
        stage_loss = combined_loss(L_ans, L_vf, L_lya, L_jac, stage=stage)
        print(f"Stage {stage} loss: {stage_loss.item()}")
        if not isinstance(stage_loss, torch.Tensor):
            all_valid = False
            
    if all_valid:
        print("Phase 6 complete")

if __name__ == "__main__":
    main()
