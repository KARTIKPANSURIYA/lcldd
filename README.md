# LCLDD (Lyapunov-based Constrained Latent Distillation)

A modular AI framework that distills teacher models' reasoning capabilities into student models through latent trajectory distillation, constrained by Lyapunov energy dynamics.

## Winning Configuration

The best performing model configuration relies on the following key techniques:

**1. Embedding-Space State Extraction**
- **`phi_x` formulation:** Extracts knowledge directly from the original input embeddings of the *last token*, rather than the mean-pooled final hidden state.
- `phi_x = model.get_input_embeddings()(input_ids).float()[:, -1, :]`

**2. Last-Token Modification Injection**
- The reasoning delta is injected exclusively by modifying the final token of the origin prompt at inference and training time.
- Prepending new "thinking" tokens was superseded by this direct modification technique.

**3. Tuned Hyperparameters**
- `phi_x_mode: "last_token_embed"`
- `injection_scale = 0.1`
- `T_max = 3` (3 discrete thinking steps)

## Usage
Run directly from Google Colab or on your local machine using the respective Python scripts:
1. `precompute_teacher.py` (extracts ground-truth trajectories).
2. `train.py` (trains the student mapping via projection head & thinking block).
3. `evaluate.py` (validates against baselines).