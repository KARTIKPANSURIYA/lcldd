import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from data.gsm8k_loader import load_gsm8k


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(CONFIG["seed"])

    print("Loading GSM8K data...")
    data = load_gsm8k(split=CONFIG["train_split"], limit=CONFIG["train_limit"])
    questions = [x["question"] for x in data]
    answers = [x["answer"] for x in data]
    print(f"Loaded {len(data)} samples from GSM8K {CONFIG['train_split']} split")

    print(f"Loading teacher model on CPU: {CONFIG['teacher_model']}")
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_model"],
        torch_dtype=torch.float32,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    ).cpu()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    proj = torch.nn.Linear(
        CONFIG["teacher_hidden_dim"],
        CONFIG["student_hidden_dim"],
        bias=False,
    )

    all_trajectories = []
    for sample in tqdm(data, desc="Precomputing teacher trajectories"):
        inputs = tokenizer(
            sample["question"],
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
        )
        with torch.no_grad():
            outputs = teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states)[-5:]
            traj = []
            for h in hidden_states:
                pooled = h.mean(dim=1).float()
                projected = proj(pooled)
                traj.append(projected.cpu())
            all_trajectories.append(traj)

    os.makedirs(os.path.dirname(CONFIG["teacher_cache"]), exist_ok=True)
    torch.save(
        {
            "trajectories": all_trajectories,
            "questions": questions,
            "answers": answers,
            "proj_state": proj.cpu().state_dict(),
            "teacher_model": CONFIG["teacher_model"],
            "student_model": CONFIG["student_model"],
            "dataset": "gsm8k",
            "split": CONFIG["train_split"],
            "train_limit": CONFIG["train_limit"],
            "teacher_hidden_dim": CONFIG["teacher_hidden_dim"],
            "student_hidden_dim": CONFIG["student_hidden_dim"],
            "T_teacher": 5,
        },
        CONFIG["teacher_cache"],
    )
    print(f"Saved teacher cache to {CONFIG['teacher_cache']}")


if __name__ == "__main__":
    main()
