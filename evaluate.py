import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.thinking_block import LyapunovThinkingBlock
import os
import re

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONFIG_HIDDEN_DIM = 1536

# ── Eval set: 10 training-domain + 5 held-out test questions ────────────
GSM8K_EVAL = [
    # --- seen during training (distribution match) ---
    {"question": "A farmer has 3 fields. Each field has 4 rows. Each row has 6 plants. How many plants total? Answer:", "answer": "72"},
    {"question": "Lisa had 120 dollars. She spent a third on books and half of what remained on food. How much left? Answer:", "answer": "40"},
    {"question": "A school has 8 classes. Each class has 25 students. 15 are absent. How many present? Answer:", "answer": "185"},
    {"question": "John walks 3 miles to school and back every day for 5 days. Total miles? Answer:", "answer": "30"},
    {"question": "A baker makes 12 loaves per hour. Works 6 hours and sells 40 loaves. How many remain? Answer:", "answer": "32"},
    {"question": "There are 5 boxes with 8 red balls and 4 blue balls each. Total balls? Answer:", "answer": "60"},
    {"question": "Maria has 3 times as many stickers as Tom. Tom has 24. Maria gives away 15. How many does Maria have? Answer:", "answer": "57"},
    {"question": "A store buys apples for 2 dollars each and sells for 3 dollars. Sells 48 apples. What is the profit? Answer:", "answer": "48"},
    {"question": "A tank holds 200 liters. It is 35 percent full. How many liters needed to fill it? Answer:", "answer": "130"},
    {"question": "A train goes 60 mph for 2 hours then 80 mph for 1 hour. Total distance? Answer:", "answer": "200"},
    # --- held-out test questions (not seen during training) ---
    {"question": "A cinema has 15 rows with 20 seats each. 47 seats are occupied. How many are empty? Answer:", "answer": "253"},
    {"question": "A worker earns 18 dollars per hour and works 7.5 hours. How much does he earn? Answer:", "answer": "135"},
    {"question": "A container has 3.5 liters of juice. If 750ml is poured out, how many ml remain? Answer:", "answer": "2750"},
    {"question": "A class of 30 students took a test. 40 percent passed. How many failed? Answer:", "answer": "18"},
    {"question": "A recipe needs 2.5 cups of flour per batch. How much flour for 4 batches? Answer:", "answer": "10"},
]


def evaluate(model, tokenizer, thinking_block, eval_data, use_thinking=True):
    correct = 0
    total = len(eval_data)
    results = []
    T_max = 5
    device = next(model.parameters()).device

    thinking_block.eval()

    for item in eval_data:
        inputs = tokenizer(
            item["question"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if use_thinking:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                phi_x = outputs.hidden_states[-1].mean(dim=1).float()

                h = phi_x.clone()
                for _ in range(T_max):
                    h = thinking_block(h, phi_x)

                original_embeds = model.get_input_embeddings()(
                    inputs["input_ids"]
                ).float()

                thinking_delta = (h - phi_x).unsqueeze(1)
                scale = original_embeds.norm(dim=-1, keepdim=True).mean()
                thinking_delta = thinking_delta * (
                    scale / (thinking_delta.norm() + 1e-8)
                )

                modified_embeds = original_embeds.clone()
                modified_embeds[:, -1:, :] = (
                    original_embeds[:, -1:, :] + 0.01 * thinking_delta
                )
                modified_embeds = modified_embeds.to(model.dtype)

                generated = model.generate(
                    inputs_embeds=modified_embeds,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            with torch.no_grad():
                generated = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

        numbers_in_output = re.findall(r'\b\d+(?:\.\d+)?\b', decoded)
        match = item["answer"] in numbers_in_output
        if match:
            correct += 1

        results.append({
            "question": item["question"],
            "expected": item["answer"],
            "predicted": decoded[:50],
            "correct": match,
        })

    print("=" * 60)
    print("LCLDD Evaluation Results")
    print(f"Mode: {'With Thinking Block' if use_thinking else 'Baseline'}")
    print("=" * 60)
    for result in results:
        status = "✅" if result["correct"] else "❌"
        print(f"{status} Expected: {result['expected']:>6} | "
              f"Got: {result['predicted'][:40]}")
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} = {correct / total * 100:.1f}%")
    print("=" * 60)
    return correct / total


if __name__ == "__main__":
    STUDENT_MODEL = "Qwen/Qwen2.5-1.5B"
    CHECKPOINT = "checkpoints/lcldd_final.pt"

    print("Loading student model (Qwen2.5-1.5B)...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        dtype=torch.float32,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading thinking block from checkpoint...")
    thinking_block = LyapunovThinkingBlock(CONFIG_HIDDEN_DIM).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        thinking_block.load_state_dict(ckpt["thinking_block_state"])
        print(f"Loaded from {CHECKPOINT}")
        if "final_losses" in ckpt:
            print(f"  Checkpoint L_lya: {ckpt['final_losses']['L_lya']:.4f}")
            print(f"  Checkpoint L_vf:  {ckpt['final_losses']['L_vf']:.4f}")
    else:
        print(f"WARNING: {CHECKPOINT} not found — using fresh (untrained) thinking block")

    print("\n--- Running BASELINE (no thinking block) ---")
    baseline = evaluate(student, tokenizer, thinking_block, GSM8K_EVAL, use_thinking=False)

    print("\n--- Running LCLDD (with thinking block injection) ---")
    lcldd = evaluate(student, tokenizer, thinking_block, GSM8K_EVAL, use_thinking=True)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline accuracy:  {baseline * 100:.1f}%")
    print(f"LCLDD accuracy:     {lcldd * 100:.1f}%")
    if lcldd > baseline:
        print(f"LCLDD improvement:  +{(lcldd - baseline) * 100:.1f}% ✅")
    elif lcldd == baseline:
        print("Same accuracy — thinking block neutral")
    else:
        print(f"LCLDD regression:   -{(baseline - lcldd) * 100:.1f}% (more training needed)")
    print("=" * 60)
