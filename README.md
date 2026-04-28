# LCLDD — Lyapunov-Constrained Latent Dynamics Distillation

## 1. Overview
LCLDD distills latent reasoning dynamics from a larger teacher model into a frozen student model through a trainable Lyapunov-constrained recursive thinking block and projection head. The repository keeps the original LCLDD concept intact (latent dynamics + Lyapunov energy constraints + injection at generation time), and now provides a reproducible GSM8K benchmark-oriented pipeline.

## 2. Current Benchmark Scope
- **Dataset:** GSM8K only.
- **Teacher:** `Qwen/Qwen2.5-3B-Instruct`.
- **Student (frozen):** `Qwen/Qwen2.5-1.5B`.

## 3. Pipeline
`precompute_teacher.py` → `train.py` → `evaluate.py`

1. **precompute_teacher.py**
   - Loads GSM8K train data.
   - Extracts the teacher hidden-state trajectory (last 5 layers).
   - Mean-pools and projects trajectory states to student hidden dimension.
   - Saves cache to disk.
2. **train.py**
   - Loads cached teacher trajectories.
   - Freezes full student model.
   - Trains only the LCLDD modules (`LyapunovThinkingBlock`, `ProjectionHead`) using answer loss + vector-field + Lyapunov constraints.
3. **evaluate.py**
   - Evaluates **baseline** and **LCLDD-injected** generation modes on GSM8K test split.
   - Computes numeric exact-match accuracy and latency.
   - Writes benchmark artifacts.

## 4. Installation
```bash
pip install -r requirements.txt
```

## 5. Quick Test Run
Set these values in `config.py`:
- `train_limit = 50`
- `eval_limit = 50`
- `num_epochs = 1`

Then run:
```bash
python precompute_teacher.py
python train.py
python evaluate.py
```

## 6. GSM8K Benchmark Run
Set these values in `config.py`:
- `train_limit = 1000`
- `eval_limit = None`
- `num_epochs = 3`

Then run:
```bash
python precompute_teacher.py
python train.py
python evaluate.py
```

## 7. Output Files
- `cache/gsm8k_teacher_train.pt`
- `checkpoints/gsm8k_lcldd_final.pt`
- `results/gsm8k_predictions.csv`
- `results/gsm8k_summary.csv`

## 8. Metrics Reported
- exact-match accuracy
- average latency
- thinking steps
- Lyapunov energy descent
- latent norm trajectory

## 9. Safe Claim Template
“On GSM8K, LCLDD achieved X% exact-match accuracy compared with Y% for the frozen Qwen2.5-1.5B baseline, while maintaining monotonic Lyapunov energy descent in Z% of evaluated samples.”
