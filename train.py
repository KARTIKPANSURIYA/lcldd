import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.thinking_block import LyapunovThinkingBlock
from models.halting import PhaseSpaceHalting
from losses.lyapunov_loss import lyapunov_loss
from losses.vf_loss import vector_field_loss
from losses.combined_loss import combined_loss
import wandb
import os
import shutil

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

CHECKPOINT_DIR = "checkpoints"
RESUME_FROM = "checkpoints/lcldd_final.pt"

STUDENT_DIM = 1536

CONFIG = {
    "student_model": "Qwen/Qwen2.5-1.5B",
    "hidden_dim": STUDENT_DIM,
    "T_max": 3,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "num_epochs": 80,
    "lambda_vf": 0.00005,
    "lambda_lya": 0.005,
    "lambda_jac": 0.0,
    "stage": "E",
    "max_length": 128,
    "device": "mps" if torch.backends.mps.is_available() else "cpu"
}

TRAIN_DATA = [
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
    {"question": "A rectangle has length 15 and width 8. What is its perimeter? Answer:", "answer": "46"},
    {"question": "A shop has 144 items split equally across 12 shelves. How many items per shelf? Answer:", "answer": "12"},
]


def load_cached_trajectories():
    cache_file = "cache/teacher_trajectories.pt"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file} - run precompute_teacher.py first!")
    cache = torch.load(cache_file, map_location="cpu")
    print(f"Loaded cached trajectories for {len(cache['trajectories'])} questions")
    return cache


def load_student_only(config):
    print("Loading student model (Qwen2.5-1.5B) on MPS...")
    student = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        dtype=torch.float32,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    ).to(config["device"])
    
    student.train()

    tokenizer = AutoTokenizer.from_pretrained(config["student_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return student, tokenizer


def compute_e2e_loss(student, tokenizer, thinking_block, batch, phi_x, student_traj, config):
    """
    Backpropagate answer cross-entropy through
    thinking block injection into embeddings.
    """
    device = config["device"]
    e2e_losses = []

    for i, item in enumerate(batch):
        # Get single sample phi_x
        phi_x_i = phi_x[i:i+1]  # shape (1, hidden_dim)

        # Get final thinking block hidden state
        h_T = student_traj[-1][i:i+1]  # shape (1, hidden_dim)

        # Tokenize question with Answer: suffix
        prompt = item["question"]
        q_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_length"]
        )
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

        # Get original embeddings - keep grad
        original_embeds = student.get_input_embeddings()(
            q_inputs["input_ids"]
        ).float()

        # Compute thinking delta
        thinking_delta = (h_T - phi_x_i).unsqueeze(1)
        scale = original_embeds.norm(dim=-1, keepdim=True).mean()
        thinking_delta = thinking_delta * (
            scale / (thinking_delta.norm() + 1e-8)
        ) * 0.01

        # Inject into last token
        modified_embeds = original_embeds.clone()
        modified_embeds[:, -1:, :] = (
            original_embeds[:, -1:, :] + thinking_delta
        )
        modified_embeds = modified_embeds.to(student.dtype)

        # Tokenize answer
        ans_ids = tokenizer(
            item["answer"],
            return_tensors="pt",
            truncation=True,
            max_length=8
        )["input_ids"].to(device)

        # Forward pass through student with modified embeds
        # This keeps gradient connection to thinking_block
        outputs = student(
            inputs_embeds=modified_embeds,
            attention_mask=q_inputs["attention_mask"],
            output_hidden_states=False
        )

        # Cross entropy on last logit vs answer first token
        logits = outputs.logits[:, -1, :]  # (1, vocab)
        target = ans_ids[:, 0]              # (1,)

        loss_i = torch.nn.functional.cross_entropy(logits, target)
        e2e_losses.append(loss_i)

    return torch.stack(e2e_losses).mean()


def train_step(batch_idx, cache, student, tokenizer, thinking_block, halter, optimizer, config, step):
    batch = [TRAIN_DATA[j] for j in batch_idx]
    inputs = tokenizer(
        [item["question"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"]
    )
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}

    # Load pre-cached teacher trajectory for this batch
    teacher_traj_per_item = []
    for idx in batch_idx:
        traj = [t.to(config["device"]).squeeze(0) for t in cache["trajectories"][idx]]
        teacher_traj_per_item.append(traj)

    # Convert from [batch_size, T, dim] to [T, batch_size, dim] for L_vf
    teacher_traj_batched = []
    for t in range(5):
        h_t = torch.stack([teacher_traj_per_item[b][t] for b in range(len(batch_idx))], dim=0)
        teacher_traj_batched.append(h_t)

    with torch.no_grad():
        student_outputs = student(
            **inputs, output_hidden_states=True
        )
        phi_x = student_outputs.hidden_states[-1].mean(dim=1)
        phi_x = phi_x.detach().to(torch.float32)

    phi_x = phi_x.detach().requires_grad_(False)
    h = phi_x.clone()
    student_traj = [h]
    for t in range(config["T_max"]):
        h = thinking_block(h, phi_x)
        student_traj.append(h)

    energy_per_step = []
    for h_state in student_traj:
        e = thinking_block.lyapunov_energy(h_state, phi_x)
        energy_per_step.append(e.mean().item())

    last_half = energy_per_step[3:]
    descending = all(
        last_half[i] >= last_half[i + 1]
        for i in range(len(last_half) - 1)
    )

    labels = tokenizer(
        [item["answer"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16
    )["input_ids"].to(config["device"])

    logits = student_outputs.logits
    L_ans = nn.CrossEntropyLoss()(logits[:, -1, :], labels[:, 0])

    L_lya = lyapunov_loss(thinking_block, student_traj, phi_x)
    teacher_traj_float = [t.float() for t in teacher_traj_batched]
    L_vf = vector_field_loss(student_traj, teacher_traj_float)

    L_vf = L_vf / (config["hidden_dim"] ** 2)
    L_lya = L_lya / config["hidden_dim"]

    L_e2e = compute_e2e_loss(
        student, tokenizer, thinking_block,
        batch, phi_x, student_traj, config
    )

    total = combined_loss(
        L_ans + 0.1 * L_e2e,
        L_vf,
        L_lya,
        torch.tensor(0.0).to(config["device"]),
        lambda_vf=config["lambda_vf"],
        lambda_lya=config["lambda_lya"],
        lambda_jac=0.0,
        stage=config["stage"]
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        thinking_block.parameters(),
        max_norm=0.1
    )
    optimizer.step()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "total": total.item(),
        "L_ans": L_ans.item(),
        "L_lya": L_lya.item(),
        "L_vf": L_vf.item(),
        "L_e2e": L_e2e.item(),
        "descending": descending,
        "energy_steps": energy_per_step
    }


if __name__ == "__main__":
    cache = load_cached_trajectories()
    student, tokenizer = load_student_only(CONFIG)

    print("Loading mapped projection layer...")
    proj = nn.Linear(2048, 1536, bias=False)
    proj.load_state_dict(cache["proj_state"])
    proj = proj.to(CONFIG["device"])

    thinking_block = LyapunovThinkingBlock(CONFIG["hidden_dim"]).to(CONFIG["device"])
    halter = PhaseSpaceHalting().to(CONFIG["device"])

    # Optimizer trains ONLY the thinking block now (projection is fixed)
    optimizer = torch.optim.AdamW(
        thinking_block.parameters(),
        lr=CONFIG["learning_rate"]
    )

    start_epoch = 0
    if os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location=CONFIG["device"])
        thinking_block.load_state_dict(ckpt["thinking_block_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resuming from {RESUME_FROM}")
    else:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        print("Starting fresh")

    print("Starting LCLDD Training...")
    print("Student:", CONFIG["student_model"])
    print("Hidden dim:", CONFIG["hidden_dim"])
    print("Stage:", CONFIG["stage"])
    print("Device:", CONFIG["device"])
    print("=" * 40)

    losses = None
    first_lya = None

    for epoch in range(CONFIG["num_epochs"]):
        for step, i in enumerate(range(0, len(TRAIN_DATA), CONFIG["batch_size"])):
            batch_idx = list(range(i, min(i + CONFIG["batch_size"], len(TRAIN_DATA))))
            losses = train_step(
                batch_idx, cache, student, tokenizer, thinking_block, halter, optimizer, CONFIG, step
            )

            if first_lya is None:
                first_lya = losses["L_lya"]

            if epoch == 0 and step == 0:
                print(f"Initial losses — L_ans:{losses['L_ans']:.4f} "
                      f"L_vf:{losses['L_vf']:.4f} L_lya:{losses['L_lya']:.4f}")

            if step % 10 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if step == 0 and (epoch + 1) % 5 == 0:
                rel = [round(e - losses['energy_steps'][0], 1)
                       for e in losses['energy_steps']]
                print(f"  Energy delta @ epoch {epoch+1}: {rel}")

            print(f"Epoch {start_epoch+epoch+1} | Step {step+1} | "
                  f"Total: {losses['total']:.4f} | "
                  f"L_lya: {losses['L_lya']:.4f} | "
                  f"L_vf: {losses['L_vf']:.4f} | "
                  f"L_e2e: {losses['L_e2e']:.4f} | "
                  f"\u2193Energy: {losses['descending']}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"lcldd_epoch_{start_epoch+epoch+1}.pt"
            )
            torch.save({
                "epoch": start_epoch + epoch,
                "thinking_block_state": thinking_block.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "losses": losses,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    if losses:
        print("=" * 40)
        print("Training complete ✅")
        print("Final L_lya:", losses["L_lya"])
        print("Final L_vf:", losses["L_vf"])
        print("Energy descending on final step:", losses["descending"])
        if losses["descending"]:
            print("Lyapunov stability achieved ✅")
        else:
            print("Lyapunov still training...")
        if first_lya is not None:
            reduction = (1 - losses["L_lya"] / first_lya) * 100
            print(f"L_lya improvement: {first_lya:.4f} -> {losses['L_lya']:.4f}")
            print(f"Reduction: {reduction:.1f}%")

        torch.save({
            "epoch": CONFIG["num_epochs"],
            "thinking_block_state": thinking_block.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": CONFIG,
            "final_losses": {
                "L_lya": losses["L_lya"],
                "L_vf": losses["L_vf"],
            },
            "energy_trajectory": losses["energy_steps"],
        }, "checkpoints/lcldd_final.pt")
        print("Final checkpoint saved: checkpoints/lcldd_final.pt")
