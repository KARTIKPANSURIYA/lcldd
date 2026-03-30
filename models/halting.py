import torch
import torch.nn as nn

class PhaseSpaceHalting(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable thresholds
        self.tau_delta   = nn.Parameter(torch.tensor(0.01))
        self.tau_energy  = nn.Parameter(torch.tensor(0.5))
        self.tau_conf    = nn.Parameter(torch.tensor(0.9))
        self.tau_entropy = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_t, h_prev, energy_t, logits):
        # 1. Computes latent change signal
        delta_t = torch.norm(h_t - h_prev, dim=-1)

        # 2. Uses Lyapunov energy signal directly
        E_t = energy_t

        # 3. Computes answer confidence signal
        probs   = torch.softmax(logits, dim=-1)
        kappa_t = probs.max(dim=-1).values

        # 4. Computes prediction entropy signal
        eta_t = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # 5. Returns halt mask as boolean tensor and signals
        halt = (
            (delta_t   < self.tau_delta)   &
            (E_t       < self.tau_energy)  &
            (kappa_t   > self.tau_conf)    &
            (eta_t     < self.tau_entropy)
        )
        
        return halt, delta_t, kappa_t, eta_t