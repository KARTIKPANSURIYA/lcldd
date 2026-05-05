# LCLDD: Lyapunov-Constrained Latent Dynamics Distillation

**Mitigating stochastic drift in small language models through stable latent-space reasoning.**

LCLDD is a research prototype for improving mathematical reasoning in frozen student language models. Instead of fine-tuning the full backbone or generating long chain-of-thought traces, LCLDD trains a lightweight recursive latent thinking block that modifies the prompt representation before answer generation.

The project was developed for GSM8K-style mathematical reasoning experiments and supports teacher trajectory precomputation, staged latent-module training, and baseline-vs-LCLDD evaluation.

---

## 1. Project Summary

Small language models can be efficient to deploy, but they often struggle with multi-step reasoning. LCLDD addresses this by learning a controlled latent update:

\[
h_{t+1} = G_\theta(h_t, \phi(x))
\]

where \(\phi(x)\) is a question-conditioned semantic anchor and \(G_\theta\) is a trainable recursive thinking block. The final latent displacement is injected into the prompt embedding before greedy answer generation.

The training objective combines:

- **Answer supervision**: encourages the latent update to improve the final numeric answer.
- **Vector-field distillation**: aligns the student's latent motion with a stronger teacher model.
- **Lyapunov-style stability regularization**: discourages uncontrolled latent drift during recursive reasoning.

---

## 2. Main Experimental Results

The main project experiments were run on the full **Grade School Math 8K (GSM8K)** test split with 1,319 examples.

| Experiment | Baseline | LCLDD | Change |
|---|---:|---:|---:|
| Qwen2.5-7B on full GSM8K | 60 / 1319 = **4.55%** | 142 / 1319 = **10.77%** | **+6.22 pts** |
| Qwen2.5-Math-7B on full GSM8K | 186 / 1319 = **14.10%** | 188 / 1319 = **14.25%** | **+0.15 pts** |
| Qwen2.5-7B 500-example ablation, Stage E | 28 / 500 = **5.60%** | 53 / 500 = **10.60%** | **+5.00 pts** |

### External 7B Reference Models

| Model | Correct | Accuracy |
|---|---:|---:|
| Qwen2.5-7B baseline | 60 / 1319 | 4.55% |
| Qwen2.5-7B-Instruct | 67 / 1319 | 5.08% |
| Mistral-7B-v0.1 | 73 / 1319 | 5.53% |
| Mistral-7B-Instruct-v0.3 | 30 / 1319 | 2.27% |
| Qwen2.5-7B + LCLDD | 142 / 1319 | 10.77% |

**Note:** External model comparisons use the same direct numeric-answer extraction format. Some instruction-tuned models may perform differently with model-specific chat templates.

---

## 3. Repository Structure

```text
.
├── config.py                    # Central configuration for models, dataset, training, and paths
├── precompute_teacher.py         # Precomputes teacher hidden-state trajectories
├── train.py                      # Trains the LCLDD thinking block and projection head
├── evaluate.py                   # Evaluates baseline and LCLDD generation on GSM8K
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license
├── CITATION.cff                  # Citation metadata for GitHub and academic reuse
├── data/
│   ├── __init__.py
│   └── gsm8k_loader.py           # GSM8K loading and numeric answer extraction
├── losses/
│   ├── combined_loss.py          # Stage-aware LCLDD loss composition
│   ├── lyapunov_loss.py          # Lyapunov energy descent penalty
│   ├── vf_loss.py                # Vector-field distillation loss used by the main training script
│   └── jac_loss.py               # Optional Stage F/G Jacobian alignment loss
└── models/
    ├── thinking_block.py         # Contractive Lyapunov thinking block
    ├── projection_head.py        # Bounded projection head for latent injection
    ├── load_models.py            # Frozen teacher/student loading helpers
    ├── vector_field.py           # Reusable vector-field utility module
    ├── jacobian_alignment.py     # Optional Jacobian alignment utility module
    └── halting.py                # Experimental dynamic halting module
```

Generated files are intentionally ignored by Git:

```text
cache/
checkpoints/
results/
wandb/
```

This keeps the repository lightweight and avoids committing large model caches, checkpoints, or experiment outputs.

---

## 4. Installation

Create a fresh Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU experiments, install a PyTorch build compatible with your CUDA version if the default installation does not match your environment.

---

## 5. Quick Start

The default code path follows this pipeline:

```bash
python precompute_teacher.py
python train.py
python evaluate.py
```

### Step 1: Precompute Teacher Trajectories

```bash
python precompute_teacher.py
```

This script:

1. Loads GSM8K training samples.
2. Runs the teacher model once over the training subset.
3. Extracts hidden states from selected upper layers.
4. Projects teacher states into the student hidden dimension.
5. Saves the result to `cache/gsm8k_teacher_train.pt`.

### Step 2: Train LCLDD

```bash
python train.py
```

This script:

1. Loads the cached teacher trajectory file.
2. Freezes the full student language model.
3. Trains only the LCLDD modules:
   - `LyapunovThinkingBlock`
   - `ProjectionHead`
4. Saves checkpoints to `checkpoints/`.

### Step 3: Evaluate

```bash
python evaluate.py
```

This script evaluates two modes:

- `baseline`: direct frozen-student generation
- `lcldd`: generation after latent displacement injection

It writes:

```text
results/gsm8k_predictions.csv
results/gsm8k_predictions.json
results/gsm8k_summary.csv
```

---

## 6. Configuration

Most settings are controlled in `config.py`.

Important fields:

| Field | Purpose |
|---|---|
| `teacher_model` | Hugging Face teacher model ID |
| `student_model` | Hugging Face student model ID |
| `teacher_hidden_dim` | Teacher hidden dimension before projection |
| `student_hidden_dim` | Student hidden dimension |
| `train_limit` | Number of GSM8K training examples for trajectory cache |
| `eval_limit` | Number of GSM8K test examples; use `None` for full test |
| `T_max` | Number of recursive latent thinking steps |
| `injection_scale` | Strength of latent displacement injection |
| `lambda_vf` | Weight for vector-field distillation |
| `lambda_lya` | Weight for Lyapunov stability loss |
| `stage` | Curriculum stage: `A`, `B`, `C`, `D`, `E`, `F`, or `G` |

### Reproducible Small Run

For a lightweight smoke test:

```python
CONFIG["train_limit"] = 50
CONFIG["eval_limit"] = 50
CONFIG["num_epochs"] = 1
```

### Full Evaluation

For full GSM8K evaluation:

```python
CONFIG["eval_limit"] = None
```

Full 7B-scale experiments require high-memory GPU hardware. The reported project runs used an A100 80GB environment.

---

## 7. Method Details

### 7.1 Latent Anchor

LCLDD extracts a question-conditioned anchor \(\phi(x)\) from the frozen student representation. This anchor keeps recursive latent updates tied to the input question.

### 7.2 Recursive Thinking Block

The thinking block applies a contractive update around the anchor:

\[
h_{t+1} = \phi(x) + (1 - \gamma)(h_t - \phi(x)) + \gamma \Delta h_t
\]

where \(\gamma\) is a learnable contraction rate and \(\Delta h_t\) is a bounded correction.

### 7.3 Lyapunov Energy

The stability energy is:

\[
V(h_t; x) = \alpha \|h_t - \phi(x)\|_2^2 + \beta \|h_t\|_2^2
\]

The Lyapunov loss penalizes energy increases across recursive steps.

### 7.4 Vector-Field Distillation

The student trajectory is trained to follow the teacher's projected direction of motion:

\[
L_{vf} = \sum_t \| (h_{t+1}^S - h_t^S) - (z_{t+1}^T - z_t^T) \|_2^2
\]

This transfers teacher reasoning direction rather than only matching final states.

---

## 8. Training Curriculum

The implementation supports a staged curriculum through `combined_loss.py`.

| Stage | Active Objective | Purpose |
|---|---|---|
| A/B/C | Answer loss | Learn task-relevant latent injection |
| D | Answer loss + vector-field loss | Align latent motion with teacher trajectory |
| E | Answer loss + vector-field loss + Lyapunov loss | Stabilize recursive latent dynamics |
| F/G | Optional Jacobian/adaptive extensions | Experimental future work |

The final evaluated method uses the core stages A, D, and E. Stage E produced the strongest 500-example ablation result in the reported experiments.

---

## 9. Evaluation Metrics

The evaluation reports:

- Numeric exact-match accuracy
- Correct/total counts
- Average latency per example
- Lyapunov energy trajectory
- Energy descent indicator
- Latent norm trajectory
- Fixed/broken example analysis when comparing runs

Answer extraction uses the final numeric value in the generated output and compares it with the normalized GSM8K gold answer.

---

## 10. Reproducibility Notes

- Set `CONFIG["seed"]` for deterministic initialization where possible.
- Use greedy decoding (`do_sample=False`) for deterministic evaluation.
- Keep teacher caches, checkpoints, and results outside Git.
- Use the same prompt format when comparing baseline and LCLDD.
- For instruction-tuned external models, model-specific chat templates may change results.
- Full 7B-scale runs are hardware-sensitive and may require A100-class GPU memory.

---

## 11. Limitations

LCLDD is a research prototype. Current limitations include:

- The default scripts are optimized for GSM8K-style numeric-answer evaluation.
- Fixed injection scales can break some baseline-correct examples.
- Stronger specialized students require smaller injection scales.
- Teacher trajectory extraction can be expensive for large teacher models.
- Dynamic halting and Jacobian alignment are implemented/planned as experimental extensions, but they are not the main evaluated result.

---

## 12. Code Availability

Repository:

```text
https://github.com/KARTIKPANSURIYA/lcldd
```

---

## 13. Citation

If you use this project, cite it as:

```bibtex
@misc{pansuriya2026lcldd,
  title  = {Mitigating Stochastic Drift in Small Language Models via Lyapunov-Constrained Latent Dynamics Distillation},
  author = {Pansuriya, Kartik and Kharwa, Hena},
  year   = {2026},
  note   = {Research project repository},
  url    = {https://github.com/KARTIKPANSURIYA/lcldd}
}
```

---

## 14. Acknowledgment

This project was developed as part of an academic research project at Stevens Institute of Technology. The authors thank the Department of Electrical and Computer Engineering for academic support and access to computing resources, and thank Prof. Min Song for project guidance and mentorship.
