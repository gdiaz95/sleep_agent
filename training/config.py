from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_token_env: str = "HF_TOKEN"          # env var that holds the HuggingFace token

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = "data/conversations.jsonl"
    train_split: float = 0.8
    val_split: float = 0.1
    # remaining 0.1 → test

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "artifacts/training/checkpoints"
    adapter_dir: str = "artifacts/training/adapter"
    merged_dir: str = "artifacts/training/merged"

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int = 64
    lora_alpha: int = 128                   # convention: alpha = 2 * r
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4    # effective batch = 16 on single GPU
    num_train_epochs: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_torch"
    save_strategy: str = "no"          # adapter is saved explicitly at the end; no mid-run checkpoints
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    logging_steps: int = 10

    # ── Evaluation ────────────────────────────────────────────────────────────
    perplexity_threshold: float = 20.0      # evaluate.py exits with code 1 above this
