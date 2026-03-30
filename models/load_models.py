from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_teacher(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        resume_download=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        local_files_only=False,
        resume_download=True,
        low_cpu_mem_usage=True
    )
    for param in model.parameters():
        param.requires_grad = False
        
    print("Teacher loaded:", model_name)
    print("Teacher params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
    return model, tokenizer

def load_student(model_name="Qwen/Qwen2.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        resume_download=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        output_hidden_states=True,
        local_files_only=False,
        resume_download=True,
        low_cpu_mem_usage=True
    )
    # Ensure all parameters track gradients for student
    for param in model.parameters():
        param.requires_grad = True
        
    print("Student loaded:", model_name)
    print("Student params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9, "B")
    return model, tokenizer

def get_hidden_states(model, tokenizer, text, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states
    last_hidden = hidden_states[-1]
    
    return hidden_states, last_hidden, inputs["input_ids"]
