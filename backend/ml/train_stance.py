#!/usr/bin/env python3
"""Full-finetune DistilBERT on SemEval stance data (baseline trainer)."""
import os, json, random, inspect
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Repository paths --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "stance_distilbert"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training config ---------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"
LABEL2ID = {"favor": 0, "against": 1, "none": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def make_dataset(path: Path) -> Dataset:
    rows = []
    for r in load_jsonl(path):
        if r.get("label") in LABEL2ID and r.get("topic") and r.get("text"):
            rows.append({"query": r["topic"], "text": r["text"], "labels": LABEL2ID[r["label"]]})
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return Dataset.from_list(rows)

def supported_kwargs(kwargs: dict):
    allowed = set(inspect.signature(TrainingArguments).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}

def build_args(output_dir: str):
    import torch
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    base = dict(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=12,     # keeps M2/CPU memory happy
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=3,      # ≈36 global batch
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=25,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,        # pinning adds no value on CPU/MPS
        gradient_checkpointing=True,
        optim="adamw_torch",
        seed=42,
        no_cuda=not torch.cuda.is_available() and not use_mps,
    )

    # Let HF auto-enable fp16/bf16 if the hardware supports it.
    base["fp16"] = True

    # Only set scheduler knobs that exist on the installed transformers.
    ta_sig = set(inspect.signature(TrainingArguments).parameters.keys())
    if "evaluation_strategy" in ta_sig:
        base.update(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            load_best_model_at_end=False,
            report_to=[]
        )
    else:
        if "do_eval" in ta_sig:
            base.update(do_eval=True)

    return TrainingArguments(**supported_kwargs(base))

def main():
    set_seeds(42)
    train_path = PROCESSED / "semeval_train.jsonl"
    dev_path   = PROCESSED / "semeval_dev.jsonl"
    assert train_path.exists(), f"Missing {train_path}"
    assert dev_path.exists(),   f"Missing {dev_path}"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    MAX_LEN = 128  # tighter window for faster training
    def tokenize(batch):
        enc = tok(batch["query"], batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = batch["labels"]
        return enc

    ds_train = make_dataset(train_path).shuffle(seed=42).map(tokenize, batched=True, remove_columns=["query","text"])
    ds_dev   = make_dataset(dev_path).map(tokenize, batched=True, remove_columns=["query","text"])

    # DistilBERT runs fine in fp16 on both CUDA and Apple MPS.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = build_args(str(MODEL_DIR))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        f1m = f1_score(p.label_ids, preds, average="macro")
        return {"accuracy": acc, "f1": f1m}

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    print(f"Saved model + tokenizer to {MODEL_DIR}")

if __name__ == "__main__":
    main()
