import torch

CONFIG = {
    "teacher_model": "Qwen/Qwen2.5-3B-Instruct",
    "student_model": "Qwen/Qwen2.5-1.5B",
    "dataset": "gsm8k",
    "train_split": "train",
    "eval_split": "test",
    # Start with small benchmark-safe default.
    # User can change to None later for full GSM8K train.
    "train_limit": 1000,
    "eval_limit": None,
    "teacher_hidden_dim": 2048,
    "student_hidden_dim": 1536,
    "T_max": 3,
    "phi_x_mode": "last_token_embed",
    "injection_scale": 0.1,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "lambda_vf": 0.01,
    "lambda_lya": 1.0,
    "lambda_jac": 0.0,
    "stage": "E",
    "max_length": 256,
    "teacher_cache": "cache/gsm8k_teacher_train.pt",
    "checkpoint_dir": "checkpoints",
    "final_checkpoint": "checkpoints/gsm8k_lcldd_final.pt",
    "results_dir": "results",
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "seed": 42,
}
