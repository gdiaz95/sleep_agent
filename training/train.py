"""
Phase 1: Single-GPU LoRA SFT on Mistral-7B-Instruct-v0.3.

Run:
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=hf_... uv run python -m training.train
"""

import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import huggingface_hub

from training.config import TrainingConfig
from training.data_prep import build_dataset


def main() -> None:
    config = TrainingConfig()

    # ── HuggingFace auth ──────────────────────────────────────────────────────
    hf_token = os.environ.get(config.hf_token_env)
    if not hf_token:
        print(f"ERROR: environment variable '{config.hf_token_env}' is not set.")
        print("Set it with:  export HF_TOKEN=hf_...")
        sys.exit(1)
    huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=hf_token,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = config.max_seq_length

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading data from: {config.data_path}")
    dataset = build_dataset(config, tokenizer)
    print(
        f"  train={len(dataset['train'])}  "
        f"val={len(dataset['validation'])}  "
        f"test={len(dataset['test'])}"
    )

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"Loading base model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        token=hf_token,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.enable_input_require_grads()      # required for LoRA + gradient checkpointing

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        fp16=config.fp16,
        optim=config.optim,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        logging_steps=config.logging_steps,
        report_to="none",                   # no wandb / tensorboard
        dataset_text_field="text",
        gradient_checkpointing=True,        # saves VRAM at the cost of ~20% speed
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"Saving adapter to: {config.adapter_dir}")
    os.makedirs(config.adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(config.adapter_dir)
    tokenizer.save_pretrained(config.adapter_dir)
    print("Done.")


if __name__ == "__main__":
    main()
