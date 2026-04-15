"""
Data pipeline: load conversations.jsonl → apply Mistral chat template → DatasetDict.
"""

import json
from pathlib import Path
from typing import List

from datasets import Dataset, DatasetDict

from training.config import TrainingConfig


def load_conversations(path: str) -> List[dict]:
    """Read JSONL file and return a list of records.

    Each record must have a 'messages' key containing a list of turns,
    e.g. [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "messages" not in record:
                raise ValueError(f"Line {line_num}: missing 'messages' key")
            if len(record["messages"]) < 2:
                raise ValueError(f"Line {line_num}: need at least 2 turns (user + assistant)")
            records.append(record)
    return records


def apply_chat_template(record: dict, tokenizer) -> dict:
    """Format one conversation into the Mistral [INST]/[/INST] string."""
    text = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_dataset(config: TrainingConfig, tokenizer) -> DatasetDict:
    """Load, format, and split conversations into train/val/test."""
    records = load_conversations(config.data_path)
    total = len(records)

    n_train = int(total * config.train_split)
    n_val = int(total * config.val_split)

    train_records = records[:n_train]
    val_records = records[n_train : n_train + n_val]
    test_records = records[n_train + n_val :]

    def format_split(recs):
        return Dataset.from_list(
            [apply_chat_template(r, tokenizer) for r in recs]
        )

    return DatasetDict({
        "train": format_split(train_records),
        "validation": format_split(val_records),
        "test": format_split(test_records),
    })
