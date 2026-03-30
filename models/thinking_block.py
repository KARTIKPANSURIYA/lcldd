import torch
import torch.nn as nn

class LyapunovThinkingBlock(nn.Module):
    def __init__(self, hidden_dim=896, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

    def lyapunov_energy(self, h, phi_x):
        # alpha * ||h - phi_x||^2 + beta * ||h||^2
        drift_term = self.alpha * torch.norm(h - phi_x, dim=-1) ** 2
        norm_term  = self.beta  * torch.norm(h, dim=-1) ** 2
        return drift_term + norm_term

    def forward(self, h_t, phi_x):
        # 1. Cross-attends h_t over phi_x
        h_attended, _ = self.cross_attn(
            h_t.unsqueeze(1), phi_x.unsqueeze(1), phi_x.unsqueeze(1)
        )
        h_attended = h_attended.squeeze(1)
        
        # 2. Concatenates h_t with attended output
        gate_input = torch.cat([h_t, h_attended], dim=-1)
        
        # 3. Passes through gated MLP to get delta_h
        delta_h = self.gate(gate_input)
        
        # 4. Returns h_t + delta_h as h_next
        h_next = h_t + delta_h
        return h_next