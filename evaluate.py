import json
import os
import random
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from data.gsm8k_loader import load_gsm8k
from models.projection_head import ProjectionHead
from models.thinking_block import LyapunovThinkingBlock


_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(x: str) -> str:
    text = (x or "").strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    return text


def extract_final_number(text: str) -> str:
    cleaned = normalize_answer(text)
    matches = _NUMBER_PATTERN.findall(cleaned)
    if not matches:
        return ""
    return normalize_answer(matches[-1])


def exact_match_numeric(prediction: str, gold: str) -> bool:
    pred_n = normalize_answer(prediction)
    gold_n = normalize_answer(gold)
    return pred_n != "" and pred_n == gold_n


def run_baseline(model, tokenizer, question: str, device: str) -> tuple[str, float]:
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output, latency_ms


def run_lcldd(model, tokenizer, question: str, thinking_block, proj_head, device: str):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    energy_steps = []
    latent_norm_steps = []
    delta_steps = []
    start = time.perf_counter()

    with torch.no_grad():
        phi_x = model.get_input_embeddings()(inputs["input_ids"]).float()[:, -1, :]

        h = phi_x.clone()
        energy_steps.append(thinking_block.lyapunov_energy(h, phi_x).mean().item())
        latent_norm_steps.append(h.norm(dim=-1).mean().item())

        for _ in range(CONFIG["T_max"]):
            h_prev = h
            h = thinking_block(h, phi_x)
            energy_steps.append(thinking_block.lyapunov_energy(h, phi_x).mean().item())
            latent_norm_steps.append(h.norm(dim=-1).mean().item())
            delta_steps.append((h - h_prev).norm(dim=-1).mean().item())

        delta = proj_head(h, phi_x)

        original_embeds = model.get_input_embeddings()(inputs["input_ids"]).float()
        embed_norm = original_embeds[:, -1, :].norm(dim=-1, keepdim=True)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * (embed_norm / (delta_norm + 1e-8)) * CONFIG["injection_scale"]

        modified_embeds = original_embeds.clone()
        modified_embeds[:, -1, :] = modified_embeds[:, -1, :] + delta

        generated = model.generate(
            inputs_embeds=modified_embeds.to(model.dtype),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000.0

    energy_descending = all(
        energy_steps[i] >= energy_steps[i + 1] for i in range(len(energy_steps) - 1)
    )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output, latency_ms, energy_steps, latent_norm_steps, delta_steps, energy_descending


def append_result(rows, i: int, mode: str, question: str, gold: str, raw_output: str, latency_ms: float,
                  thinking_steps: int, energy_descending: Optional[bool], energy_steps, latent_norm_steps, delta_steps):
    prediction = extract_final_number(raw_output)
    rows.append(
        {
            "id": i,
            "mode": mode,
            "question": question,
            "gold": normalize_answer(gold),
            "prediction": prediction,
            "raw_output": raw_output,
            "correct": exact_match_numeric(prediction, gold),
            "latency_ms": latency_ms,
            "thinking_steps": thinking_steps,
            "energy_descending": energy_descending,
            "energy_steps": json.dumps(energy_steps) if energy_steps is not None else None,
            "latent_norm_steps": json.dumps(latent_norm_steps) if latent_norm_steps is not None else None,
            "delta_steps": json.dumps(delta_steps) if delta_steps is not None else None,
        }
    )


def main() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(CONFIG["seed"])

    device = CONFIG["device"]
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    eval_data = load_gsm8k(split=CONFIG["eval_split"], limit=CONFIG["eval_limit"])
    print(f"Loaded {len(eval_data)} evaluation samples")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["student_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        CONFIG["student_model"],
        torch_dtype=torch.float32,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    ).to(device)
    student.eval()

    thinking_block = LyapunovThinkingBlock(CONFIG["student_hidden_dim"]).to(device)
    proj_head = ProjectionHead(CONFIG["student_hidden_dim"]).to(device)

    if os.path.exists(CONFIG["final_checkpoint"]):
        ckpt = torch.load(CONFIG["final_checkpoint"], map_location=device)
        thinking_block.load_state_dict(ckpt["thinking_block_state"])
        proj_head.load_state_dict(ckpt["proj_head_state"])
        print(f"Loaded LCLDD checkpoint: {CONFIG['final_checkpoint']}")
    else:
        print(f"Checkpoint not found: {CONFIG['final_checkpoint']} (running with untrained LCLDD modules)")

    rows = []
    for i, sample in enumerate(eval_data):
        raw_b, lat_b = run_baseline(student, tokenizer, sample["question"], device)
        append_result(
            rows,
            i,
            "baseline",
            sample["question"],
            sample["answer"],
            raw_b,
            lat_b,
            0,
            None,
            None,
            None,
            None,
        )

        raw_l, lat_l, energy, latent_norms, deltas, descending = run_lcldd(
            student, tokenizer, sample["question"], thinking_block, proj_head, device
        )
        append_result(
            rows,
            i,
            "lcldd",
            sample["question"],
            sample["answer"],
            raw_l,
            lat_l,
            CONFIG["T_max"],
            descending,
            energy,
            latent_norms,
            deltas,
        )

    pred_df = pd.DataFrame(rows)
    summary_df = (
        pred_df.groupby("mode", as_index=False)
        .agg(correct=("correct", "sum"), total=("correct", "count"), avg_latency_ms=("latency_ms", "mean"))
    )
    summary_df["accuracy"] = summary_df["correct"] / summary_df["total"]
    summary_df = summary_df[["mode", "correct", "total", "accuracy", "avg_latency_ms"]]

    pred_csv = os.path.join(CONFIG["results_dir"], "gsm8k_predictions.csv")
    pred_json = os.path.join(CONFIG["results_dir"], "gsm8k_predictions.json")
    summary_csv = os.path.join(CONFIG["results_dir"], "gsm8k_summary.csv")

    pred_df.to_csv(pred_csv, index=False)
    pred_df.to_json(pred_json, orient="records", indent=2)
    summary_df.to_csv(summary_csv, index=False)

    print("\nFinal comparison table:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {pred_csv}")
    print(f"Saved: {pred_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
