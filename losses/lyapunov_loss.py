import torch
import torch.nn.functional as F

def lyapunov_loss(thinking_block, trajectory, phi_x, epsilon=1e-3):
    total_loss = 0.0
    # sum over t of: max(0, V(h_{t+1}) - V(h_t) + epsilon)
    for t in range(len(trajectory) - 1):
        V_t   = thinking_block.lyapunov_energy(trajectory[t],   phi_x)
        V_t1  = thinking_block.lyapunov_energy(trajectory[t+1], phi_x)
        # We compute the mean loss
        total_loss += F.relu(V_t1 - V_t + epsilon).mean()
        
    return total_loss / max(1, len(trajectory) - 1)