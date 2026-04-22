"""
LyapunovThinkingBlock — corrected version for LCLDD.

Contractive dynamical system:

    h_{t+1} = phi_x + (1 - gamma) * (h_t - phi_x) + gamma * delta_h

where
    - phi_x       : semantic anchor (fixed point of the dynamics)
    - gamma in (0,1): learnable scalar contraction rate
    - delta_h     : bounded correction from gated MLP + cross-attention

Energy used by lyapunov_loss:

    V(h; x) = alpha * ||h - phi_x||^2 + beta * ||h||^2

Why the contraction matters
---------------------------
With the drift component e_t := h_t - phi_x, the update gives

    e_{t+1} = (1 - gamma) * e_t + gamma * delta_h

so ||e|| contracts geometrically at rate (1 - gamma) per step whenever
delta_h is bounded. This is what makes Lyapunov descent feasible.

What changed vs. the original
-----------------------------
1. Added the (1 - gamma) * (h_t - phi_x) contraction term. The old form
   `h_next = h_t + delta_h` could only grow, since delta_h was forced to
   unit variance per dim by the trailing LayerNorm (see 2).
2. Replaced the trailing `nn.LayerNorm(hidden_dim)` in `self.gate` with
   `tanh(...)` in `forward`. LayerNorm forced ||delta_h|| ~ sqrt(d)
   regardless of input; tanh lets delta_h shrink when the gate learns
   to output small pre-activations, which is the path the Lyapunov
   loss needs.
3. gamma starts small (sigmoid(-2) ~= 0.12) so Proposition 1 is
   approximately satisfied at init — energy descends before any training.
"""
import torch
import torch.nn as nn


class LyapunovThinkingBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        alpha=1.0,
        beta=0.1,
        num_heads=8,
        delta_scale=1.0,
        init_gamma=-2.0,   # sigmoid(-2) ~= 0.12
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.alpha       = alpha
        self.beta        = beta
        self.delta_scale = delta_scale

        # Learnable scalar contraction rate in (0, 1).
        self.gamma_logit = nn.Parameter(torch.tensor(float(init_gamma)))

        # Gated MLP — no trailing LayerNorm.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention of h_t over phi_x.
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

    # --------------------------------------------------------------
    # Energy function
    # --------------------------------------------------------------
    def lyapunov_energy(self, h, phi_x):
        """V(h; x) = alpha * ||h - phi_x||^2 + beta * ||h||^2, per-batch."""
        drift = self.alpha * (h - phi_x).pow(2).sum(dim=-1)
        norm_ = self.beta  * h.pow(2).sum(dim=-1)
        return drift + norm_

    @property
    def gamma(self):
        """Current contraction rate — useful for logging."""
        return torch.sigmoid(self.gamma_logit)

    # --------------------------------------------------------------
    # Forward — one recursive step
    # --------------------------------------------------------------
    def forward(self, h_t, phi_x):
        # 1. Cross-attention over phi_x (single-token Q/K/V).
        h_attn, _ = self.cross_attn(
            h_t.unsqueeze(1),
            phi_x.unsqueeze(1),
            phi_x.unsqueeze(1),
        )
        h_attn = h_attn.squeeze(1)

        # 2. Bounded delta via tanh (replaces the old LayerNorm).
        gate_in = torch.cat([h_t, h_attn], dim=-1)
        delta   = self.delta_scale * torch.tanh(self.gate(gate_in))

        # 3. Contractive update.
        gamma  = torch.sigmoid(self.gamma_logit)
        h_next = phi_x + (1.0 - gamma) * (h_t - phi_x) + gamma * delta
        return h_next