import torch
from models.thinking_block import LyapunovThinkingBlock
from models.halting import PhaseSpaceHalting
from losses.lyapunov_loss import lyapunov_loss
from losses.vf_loss import vector_field_loss
from losses.jac_loss import jacobian_loss
from losses.combined_loss import combined_loss

def main():
    # 2. Setup
    hidden_dim = 256
    batch_size = 4
    T_max = 10
    vocab_size = 100

    # 3. Initialize
    thinking_block = LyapunovThinkingBlock(hidden_dim)
    halter = PhaseSpaceHalting()

    # 4. Create fake data
    # Create x_embed BEFORE the recursive loop to seed phi_x
    x_embed = torch.randn(batch_size, 10, hidden_dim, requires_grad=True)
    # Using mean to give a true differentiable connection from x_embed to phi_x
    phi_x = x_embed.mean(dim=1)
    
    teacher_trajectory = [torch.randn(batch_size, hidden_dim) for _ in range(T_max + 1)]

    # 5. Run student recursive loop with dynamic halting
    h = phi_x.clone()
    student_trajectory = [h]
    halt_steps = torch.zeros(batch_size)
    
    for t in range(T_max):
        h_prev = h
        h = thinking_block(h, phi_x)
        student_trajectory.append(h)
        
        energy_t = thinking_block.lyapunov_energy(h, phi_x)
        fake_logits = torch.randn(batch_size, vocab_size)
        
        halt, delta, conf, ent = halter(h, h_prev, energy_t, fake_logits)
        
        # For any sample where halt=True and halt_steps=0, record t+1
        unhalted_mask = (halt_steps == 0)
        newly_halted = halt & unhalted_mask
        halt_steps[newly_halted] = t + 1

    # Print average halt step (or T_max if never halted)
    never_halted = (halt_steps == 0)
    halt_steps[never_halted] = T_max
    
    avg_halt_step = halt_steps.float().mean().item()

    # 6. Compute all losses at Stage F
    L_ans = torch.tensor(2.5, requires_grad=True)
    L_lya = lyapunov_loss(thinking_block, student_trajectory, phi_x)
    L_vf  = vector_field_loss(student_trajectory, teacher_trajectory)
    
    # passed through a simple linear layer so gradients connect
    linear = torch.nn.Linear(hidden_dim, hidden_dim)
    h_T_s = linear(student_trajectory[-1])
    h_T_t = teacher_trajectory[-1].requires_grad_(True)
    
    L_jac = jacobian_loss(h_T_s, h_T_t, x_embed)
    
    total = combined_loss(L_ans, L_vf, L_lya, L_jac, stage='F')

    # 7. Run backward pass
    total.backward()
    print("Backward pass successful")

    # 8. Print full summary
    print("=== LCLDD Smoke Test ===")
    print(f"L_ans:  {L_ans.item():.4f}")
    print(f"L_lya:  {L_lya.item():.4f}")
    print(f"L_vf:   {L_vf.item():.4f}")
    print(f"L_jac:  {L_jac.item():.4f}")
    print(f"Total:  {total.item():.4f}")
    print(f"Avg halt step: {avg_halt_step} / {T_max}")
    print("All components integrated successfully ✅")

if __name__ == "__main__":
    main()
