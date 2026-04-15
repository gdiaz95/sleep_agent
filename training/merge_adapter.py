"""
Phase 3a: Merge the LoRA adapter into the base model weights.

Produces a standard HuggingFace model directory at artifacts/training/merged/
that can be converted to GGUF for Ollama import.

Run:
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=hf_... uv run python -m training.merge_adapter
"""

import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub

from training.config import TrainingConfig


def main() -> None:
    config = TrainingConfig()

    # ── Auth ──────────────────────────────────────────────────────────────────
    hf_token = os.environ.get(config.hf_token_env)
    if not hf_token:
        print(f"ERROR: '{config.hf_token_env}' env var not set.")
        sys.exit(1)
    huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from: {config.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(config.adapter_dir, use_fast=True)

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"Loading base model: {config.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="cpu",           # merge on CPU — avoids VRAM limit for the full model copy
        token=hf_token,
    )

    # ── Attach and merge adapter ───────────────────────────────────────────────
    print(f"Attaching adapter from: {config.adapter_dir}")
    model = PeftModel.from_pretrained(base, config.adapter_dir)

    print("Merging adapter into base weights (this takes a minute)...")
    merged = model.merge_and_unload()   # returns a plain transformers model, no LoRA wrappers

    # ── Save merged model ─────────────────────────────────────────────────────
    print(f"Saving merged model to: {config.merged_dir}")
    os.makedirs(config.merged_dir, exist_ok=True)
    merged.save_pretrained(config.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(config.merged_dir)

    print(f"\nDone. Merged model saved to: {config.merged_dir}")
    print("Next step: convert to GGUF with llama.cpp")


if __name__ == "__main__":
    main()
