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

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG = {
    "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "student_model": "Qwen/Qwen2.5-0.5B",
    "hidden_dim": 896,
    "T_max": 5,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "num_epochs": 20,
    "lambda_vf": 0.0001,
    "lambda_lya": 0.005,
    "lambda_jac": 0.0,
    "stage": "E",
    "max_length": 64,
    "device": "mps" if torch.backends.mps.is_available() else "cpu"
}

TRAIN_DATA = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is 5 * 3?", "answer": "15"},
    {"question": "What is 10 - 7?", "answer": "3"},
    {"question": "What is 12 / 4?", "answer": "3"},
    {"question": "What is 8 + 9?", "answer": "17"},
    {"question": "What is 6 * 7?", "answer": "42"},
    {"question": "What is 100 - 45?", "answer": "55"},
    {"question": "What is 3 * 3?", "answer": "9"},
]

def load_models(config):
    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"],
        torch_dtype=torch.float16,
        output_hidden_states=True,
        resume_download=True,
        low_cpu_mem_usage=True
    ).to(config["device"])
    
    for p in teacher.parameters():
        p.requires_grad = False
        
    teacher.eval()
    
    student = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        torch_dtype=torch.float32,
        output_hidden_states=True,
        resume_download=True,
        low_cpu_mem_usage=True
    ).to(config["device"])
    
    student.train()
    
    tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return teacher, student, tokenizer

def get_trajectory(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return list(outputs.hidden_states)

def train_step(batch, teacher, student, thinking_block, halter, optimizer, config, step):
    inputs = tokenizer(
        [item["question"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"]
    )
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}
    
    with torch.no_grad():
        teacher_hidden = get_trajectory(
            teacher, inputs["input_ids"], inputs["attention_mask"]
        )
        teacher_traj = list(teacher_hidden[-5:])
        teacher_traj = [h.mean(dim=1) for h in teacher_traj]
        
    student_outputs = student(
        **inputs, output_hidden_states=True
    )
    phi_x = student_outputs.hidden_states[-1].mean(dim=1)
    phi_x = phi_x.to(torch.float32)
    
    h = phi_x.clone()
    student_traj = [h]
    for t in range(config["T_max"]):
        h = thinking_block(h, phi_x)
        student_traj.append(h)
        
    energy_per_step = []
    for h_state in student_traj:
        e = thinking_block.lyapunov_energy(h_state, phi_x)
        energy_per_step.append(e.mean().item())

    avg_first_half = sum(energy_per_step[:3]) / 3
    avg_second_half = sum(energy_per_step[2:]) / 3
    descending = avg_second_half < avg_first_half
    
    labels = tokenizer(
        [item["answer"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16
    )["input_ids"].to(config["device"])
    
    logits = student_outputs.logits
    L_ans = nn.CrossEntropyLoss()(
        logits[:, -1, :],
        labels[:, 0]
    )
    
    L_lya = lyapunov_loss(thinking_block, student_traj, phi_x)
    
    teacher_traj_float = [t.float() for t in teacher_traj]
    L_vf = vector_field_loss(student_traj, teacher_traj_float)
    
    L_vf = L_vf / (config["hidden_dim"] ** 2)
    L_lya = L_lya / config["hidden_dim"]
    
    # Direct energy minimization at each step
    direct_lya = 0.0
    for h_state in student_traj[1:]:
        energy = thinking_block.lyapunov_energy(h_state, phi_x)
        direct_lya += energy.mean()
    direct_lya = direct_lya / len(student_traj[1:])
    direct_lya = direct_lya / config["hidden_dim"]
    
    total = combined_loss(
        L_ans + 0.001 * direct_lya,
        L_vf, L_lya,
        torch.tensor(0.0).to(config["device"]),
        lambda_vf=config["lambda_vf"],
        lambda_lya=config["lambda_lya"],
        lambda_jac=0.0,
        stage=config["stage"]
    )
    
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        thinking_block.parameters(), max_norm=0.1
    )
    optimizer.step()
    
    return {
        "total": total.item(),
        "L_ans": L_ans.item(),
        "L_lya": L_lya.item(),
        "L_vf": L_vf.item(),
        "descending": descending,
        "energy_steps": energy_per_step
    }

if __name__ == "__main__":
    teacher, student, tokenizer = load_models(CONFIG)
    
    thinking_block = LyapunovThinkingBlock(
        CONFIG["hidden_dim"]
    ).to(CONFIG["device"])
    
    halter = PhaseSpaceHalting().to(CONFIG["device"])
    
    optimizer = torch.optim.AdamW(
        thinking_block.parameters(),
        lr=CONFIG["learning_rate"]
    )
    
    print("Starting LCLDD Training...")
    print("Stage:", CONFIG["stage"])
    print("Device:", CONFIG["device"])
    print("="*40)
    
    losses = None
    first_lya = None
    
    for epoch in range(CONFIG["num_epochs"]):
        for step, i in enumerate(
            range(0, len(TRAIN_DATA), CONFIG["batch_size"])
        ):
            batch = TRAIN_DATA[i:i+CONFIG["batch_size"]]
            losses = train_step(batch, teacher, student,
                thinking_block, halter, optimizer, CONFIG, step)
                
            if first_lya is None:
                first_lya = losses["L_lya"]
                
            if epoch == 0 and step == 0:
                print(f"Initial losses — L_ans:{losses['L_ans']:.4f} "
                      f"L_vf:{losses['L_vf']:.4f} L_lya:{losses['L_lya']:.4f}")
                
            if (epoch == 0 and step == 0) or (epoch == 19 and step == 0):
                print(f"Energy trajectory: {[round(e,2) for e in losses['energy_steps']]}")
                
            print(f"Epoch {epoch+1} | Step {step+1} | "
                  f"Total: {losses['total']:.4f} | "
                  f"L_ans: {losses['L_ans']:.4f} | "
                  f"L_lya: {losses['L_lya']:.4f} | "
                  f"L_vf: {losses['L_vf']:.4f} | "
                  f"Energy descending: {losses['descending']}")

    if losses:
        print("="*40)
        print("Training complete ✅")
        print("Final L_lya:", losses["L_lya"])
        print("Final L_vf:", losses["L_vf"])
        print("Energy descending on final step:", losses["descending"])
        if losses["descending"]:
            print("Lyapunov stability achieved ✅")
        else:
            print("Lyapunov still training...")
