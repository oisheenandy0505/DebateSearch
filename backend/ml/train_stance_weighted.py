#!/usr/bin/env python3
import os, json, random, inspect
from pathlib import Path
import numpy as np
from collections import Counter
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
PRO  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "stance_distilbert"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "distilbert-base-uncased"
LABEL2ID = {"favor":0, "against":1, "none":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.0")

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_jsonl(p: Path):
    with p.open() as f:
        for line in f: yield json.loads(line)

def make_ds(p: Path) -> Dataset:
    rows=[]
    for r in load_jsonl(p):
        if r.get("label") in LABEL2ID and r.get("topic") and r.get("text"):
            rows.append({"query": r["topic"], "text": r["text"], "labels": LABEL2ID[r["label"]]})
    if not rows: raise ValueError(f"No usable rows in {p}")
    return Dataset.from_list(rows)

def supported_kwargs(d: dict):
    allowed=set(inspect.signature(TrainingArguments).parameters.keys())
    return {k:v for k,v in d.items() if k in allowed}

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=None)  # lazily moves weights to device

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.get("logits")

        if self.class_weights is not None and self.loss_fct.weight is None:
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    set_seeds(42)
    train_path = PRO/"semeval_train.jsonl"
    dev_path   = PRO/"semeval_dev.jsonl"
    assert train_path.exists() and dev_path.exists()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    MAX_LEN = 192

    def tokenize(b):
        enc = tok(b["query"], b["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = b["labels"]
        return enc

    ds_tr_raw = make_ds(train_path)
    ds_de_raw = make_ds(dev_path)

    # Inverse-frequency weights keep the macro F1 aligned with imbalanced labels.
    cnt = Counter(ds_tr_raw["labels"])
    total = sum(cnt.values())
    weights = []
    for lab in [0,1,2]:
        freq = max(1, cnt.get(lab, 1))
        weights.append(total / (3.0 * freq))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    ds_tr = ds_tr_raw.shuffle(seed=42).map(tokenize, batched=True, remove_columns=["query","text"])
    ds_de = ds_de_raw.map(tokenize, batched=True, remove_columns=["query","text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    base_args = dict(
        output_dir=str(MODEL_DIR),
        learning_rate=3e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=3,  # ≈36 global batch
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=25,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        fp16=True,
        seed=42,
        no_cuda=not torch.cuda.is_available() and not use_mps,
    )
    if "evaluation_strategy" in inspect.signature(TrainingArguments).parameters:
        base_args.update(evaluation_strategy="epoch", save_strategy="epoch", report_to=[])

    args = TrainingArguments(**supported_kwargs(base_args))

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro")
        }

    trainer = WeightedTrainer(
        model=model,
        args=args,
        tokenizer=tok,              # deprecation warning is fine
        train_dataset=ds_tr,
        eval_dataset=ds_de,
        compute_metrics=metrics,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    print(f"Saved weighted model to {MODEL_DIR}")

if __name__ == "__main__":
    main()
