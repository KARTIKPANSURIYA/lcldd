import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading teacher model (Qwen2.5-3B-Instruct) on CPU...")
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float32,
    output_hidden_states=True,
    low_cpu_mem_usage=True
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Teacher loaded on CPU")

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

proj = torch.nn.Linear(2048, 1536, bias=False)
all_trajectories = []

print("Extracting trajectories...")
for i, d in enumerate(TRAIN_DATA):
    print(f"Processing question {i+1}/12...")
    inputs = tokenizer(
        d["question"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = teacher(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        hidden_states = list(outputs.hidden_states)[-5:]
        traj = []
        for h in hidden_states:
            h_mean = h.mean(dim=1).float()
            h_proj = proj(h_mean)
            traj.append(h_proj)
        all_trajectories.append(traj)

os.makedirs("cache", exist_ok=True)
torch.save({
    "trajectories": all_trajectories,
    "questions": [d["question"] for d in TRAIN_DATA],
    "answers": [d["answer"] for d in TRAIN_DATA],
    "proj_state": proj.state_dict(),
    "teacher_model": "Qwen/Qwen2.5-3B-Instruct",
    "hidden_dim": 1536,
}, "cache/teacher_trajectories.pt")

print("Saved cache/teacher_trajectories.pt")
print(f"Cached {len(all_trajectories)} trajectories")
print(f"Each trajectory: 5 steps x shape {all_trajectories[0][0].shape}")
print("Done! Teacher model can now be unloaded.")
print("Run train.py to start training without teacher in memory.")
