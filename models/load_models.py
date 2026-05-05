"""
Model loading helpers for LCLDD.

The final LCLDD pipeline keeps both teacher and student language-model
backbones frozen. Only the lightweight LCLDD modules, such as the thinking block
and projection head, are trained. These helpers are kept for debugging and small
experiments; the main scripts load models directly from their configured model
IDs.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_device(preferred: str = "cuda") -> str:
    """Return the best available device string."""
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_pad_token(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Ensure tokenizer has a pad token for batched inference."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def freeze_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Freeze all parameters and set the model to evaluation mode."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def count_parameters(model: AutoModelForCausalLM) -> tuple[float, float]:
    """Return total and trainable parameter counts in billions."""
    total = sum(p.numel() for p in model.parameters()) / 1e9
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    return total, trainable


def load_teacher(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a frozen teacher model and tokenizer.

    The teacher is used to extract latent trajectories during precomputation. It
    is not trained in the LCLDD pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    tokenizer = ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        output_hidden_states=True,
        local_files_only=False,
        low_cpu_mem_usage=True,
    )
    model = freeze_model(model)

    total_params, trainable_params = count_parameters(model)
    print(f"Teacher loaded: {model_name}")
    print(f"Teacher params: {total_params:.3f}B")
    print(f"Teacher trainable params: {trainable_params:.3f}B")
    return model, tokenizer


def load_student(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    torch_dtype: torch.dtype = torch.float32,
    device_map: str = "auto",
    freeze_backbone: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a student model and tokenizer.

    By default, the student backbone is frozen because LCLDD trains only the
    thinking block and projection head. Set ``freeze_backbone=False`` only for
    separate full-fine-tuning experiments outside the main LCLDD setting.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    tokenizer = ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        output_hidden_states=True,
        local_files_only=False,
        low_cpu_mem_usage=True,
    )

    if freeze_backbone:
        model = freeze_model(model)
    else:
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    total_params, trainable_params = count_parameters(model)
    print(f"Student loaded: {model_name}")
    print(f"Student params: {total_params:.3f}B")
    print(f"Student trainable params: {trainable_params:.3f}B")
    return model, tokenizer


def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 512,
):
    """Return hidden states, final hidden state, and token IDs for a text input."""
    resolved_device = resolve_device(device)

    if not hasattr(model, "hf_device_map"):
        model = model.to(resolved_device)

    tokenizer = ensure_pad_token(tokenizer)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    last_hidden = hidden_states[-1]
    return hidden_states, last_hidden, inputs["input_ids"]
