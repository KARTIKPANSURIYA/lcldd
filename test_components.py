import torch
from models.thinking_block import LyapunovThinkingBlock
from models.halting import PhaseSpaceHalting
from losses.lyapunov_loss import lyapunov_loss
from losses.vf_loss import vector_field_loss

d = 256
B = 4
T = 5

block   = LyapunovThinkingBlock(d)
halter  = PhaseSpaceHalting()

# Fake trajectory
phi_x  = torch.randn(B, d)
h      = phi_x.clone()
traj_s = [h]

for t in range(T):
    h = block(h, phi_x)
    traj_s.append(h)

traj_t = [torch.randn(B, d) for _ in range(T+1)]

L_lya = lyapunov_loss(block, traj_s, phi_x)
L_vf  = vector_field_loss(traj_s, traj_t)

print(f"L_lya: {L_lya.item():.4f}")
print(f"L_vf:  {L_vf.item():.4f}")
print("All components working ✅")
