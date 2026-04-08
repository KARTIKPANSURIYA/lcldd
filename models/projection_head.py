import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # bound output to [-1, 1]
        )
        # Start with near-zero output so it doesn't
        # disrupt generation at init
        nn.init.zeros_(self.proj[-2].weight)
        nn.init.zeros_(self.proj[-2].bias)

    def forward(self, h_T, phi_x):
        # delta = learned mapping from thinking to embedding space
        delta = self.proj(h_T - phi_x)
        return delta
