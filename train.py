import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from losses.combined_loss import combined_loss
from losses.lyapunov_loss import lyapunov_loss
from losses.vf_loss import vector_field_loss
from models.halting import PhaseSpaceHalting
from models.projection_head import ProjectionHead
from models.thinking_block import LyapunovThinkingBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_energy_descending(thinking_block, student_traj, phi_x) -> tuple[list[float], bool]:
    energy_steps = [
        thinking_block.lyapunov_energy(h_t, phi_x).mean().item() for h_t in student_traj
    ]
    descending = all(
        energy_steps[i] >= energy_steps[i + 1] for i in range(len(energy_steps) - 1)
    )
    return energy_steps, descending


def main() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    if not os.path.exists(CONFIG["teacher_cache"]):
        raise FileNotFoundError(
            f"Missing cache file: {CONFIG['teacher_cache']}. Run precompute_teacher.py first."
        )

    cache = torch.load(CONFIG["teacher_cache"], map_location="cpu")
    questions = cache["questions"]
    answers = cache["answers"]

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
    for p in student.parameters():
        p.requires_grad = False

    thinking_block = LyapunovThinkingBlock(CONFIG["student_hidden_dim"]).to(device)
    proj_head = ProjectionHead(CONFIG["student_hidden_dim"]).to(device)
    halter = PhaseSpaceHalting().to(device)
    _ = halter

    optimizer = torch.optim.AdamW(
        list(thinking_block.parameters()) + list(proj_head.parameters()),
        lr=CONFIG["learning_rate"],
    )
    ce_loss = CrossEntropyLoss()

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    final_losses = {}
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        running = {"total": 0.0, "L_ans": 0.0, "L_e2e": 0.0, "L_vf": 0.0, "L_lya": 0.0}

        pbar = tqdm(range(len(questions)), desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
        for i in pbar:
            batch_questions = [questions[i]]
            batch_answers = [answers[i]]

            inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_length"],
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                student_outputs = student(**inputs, output_hidden_states=True)

            if CONFIG["phi_x_mode"] == "last_token_embed":
                phi_x = student.get_input_embeddings()(inputs["input_ids"]).float()[:, -1, :]
            else:
                phi_x = student_outputs.hidden_states[-1].mean(dim=1).float()
            phi_x = phi_x.detach()

            h = phi_x.clone()
            student_traj = [h]
            for _ in range(CONFIG["T_max"]):
                h = thinking_block(h, phi_x)
                student_traj.append(h)

            teacher_traj_batched = [t.to(device).float() for t in cache["trajectories"][i]]

            L_lya = lyapunov_loss(thinking_block, student_traj, phi_x)
            L_vf = vector_field_loss(student_traj, teacher_traj_batched)

            ans_tokens = tokenizer(
                batch_answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8,
            )["input_ids"].to(device)
            answer_first_token = ans_tokens[:, 0]

            L_ans = ce_loss(student_outputs.logits[:, -1, :], answer_first_token)

            original_embeds = student.get_input_embeddings()(inputs["input_ids"]).float()
            h_T = student_traj[-1]
            delta = proj_head(h_T, phi_x)
            embed_norm = original_embeds[:, -1, :].norm(dim=-1, keepdim=True)
            delta_norm = delta.norm(dim=-1, keepdim=True)
            delta = delta * (embed_norm / (delta_norm + 1e-8)) * CONFIG["injection_scale"]

            modified_embeds = original_embeds.clone()
            modified_embeds[:, -1, :] = modified_embeds[:, -1, :] + delta

            injected_outputs = student(
                inputs_embeds=modified_embeds.to(student.dtype),
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            )
            L_e2e = ce_loss(injected_outputs.logits[:, -1, :], answer_first_token)

            total = combined_loss(
                L_ans + 0.5 * L_e2e,
                L_vf,
                L_lya,
                torch.tensor(0.0, device=device),
                lambda_vf=CONFIG["lambda_vf"],
                lambda_lya=CONFIG["lambda_lya"],
                lambda_jac=CONFIG["lambda_jac"],
                stage=CONFIG["stage"],
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(thinking_block.parameters()) + list(proj_head.parameters()),
                max_norm=0.1,
            )
            optimizer.step()

            energy_steps, descending = compute_energy_descending(
                thinking_block, student_traj, phi_x
            )

            final_losses = {
                "total": float(total.item()),
                "L_ans": float(L_ans.item()),
                "L_e2e": float(L_e2e.item()),
                "L_vf": float(L_vf.item()),
                "L_lya": float(L_lya.item()),
                "energy_descending": bool(descending),
                "energy_steps": energy_steps,
            }

            for key in ["total", "L_ans", "L_e2e", "L_vf", "L_lya"]:
                running[key] += final_losses[key]

            pbar.set_postfix(
                total=f"{final_losses['total']:.4f}",
                L_ans=f"{final_losses['L_ans']:.4f}",
                L_e2e=f"{final_losses['L_e2e']:.4f}",
                L_vf=f"{final_losses['L_vf']:.4f}",
                L_lya=f"{final_losses['L_lya']:.4f}",
                descend=str(final_losses["energy_descending"]),
            )

        epoch_ckpt = os.path.join(
            CONFIG["checkpoint_dir"], f"gsm8k_lcldd_epoch_{epoch}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "thinking_block_state": thinking_block.state_dict(),
                "proj_head_state": proj_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": CONFIG,
                "losses": final_losses,
            },
            epoch_ckpt,
        )

        avg = {k: v / len(questions) for k, v in running.items()}
        print(
            f"[Epoch {epoch}] total={avg['total']:.4f} L_ans={avg['L_ans']:.4f} "
            f"L_e2e={avg['L_e2e']:.4f} L_vf={avg['L_vf']:.4f} L_lya={avg['L_lya']:.4f} "
            f"Lyapunov descending={final_losses.get('energy_descending', False)}"
        )

    torch.save(
        {
            "epoch": CONFIG["num_epochs"],
            "thinking_block_state": thinking_block.state_dict(),
            "proj_head_state": proj_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": CONFIG,
            "losses": final_losses,
        },
        CONFIG["final_checkpoint"],
    )
    print(f"Saved final checkpoint to {CONFIG['final_checkpoint']}")


if __name__ == "__main__":
    main()
