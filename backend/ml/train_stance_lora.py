#!/usr/bin/env python3
"""
LoRA trainer — weighted loss + speed knobs for Mac M2.

Matches train_stance_weighted functionality (class weighting, (query,text),
macro-F1) but trains adapters only and can run in 'fast' modes.

ENV KNOBS (all optional):
  MAX_LEN=128          # token length (matches API via STANCE_MAX_LEN_INFER)
  EPOCHS=2             # small for dev; 3 for full
  MAX_STEPS=0          # e.g., 120 for a smoke run; 0 = unlimited
  SUBSAMPLE_PER_CLASS=0# e.g., 1000 keeps 1k rows per class; 0 = no cap
  LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.05
  LR=2e-4              # higher OK for adapters
  BATCH_TRAIN=12 BATCH_EVAL=24
  GRAD_ACCUM=1         # 1 on M2 is usually fastest
  PATIENCE=2           # early stopping patience (epochs)
"""

import os, json, random, inspect
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from peft import LoraConfig, TaskType, get_peft_model

# Paths & runtime knobs --------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PRO  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "stance_distilbert"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = os.getenv("BASE_MODEL", "distilbert-base-uncased")
LABEL2ID = {"favor":0, "against":1, "none":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

MAX_LEN   = int(os.getenv("MAX_LEN", "128"))   # shorter window keeps adapters quick
EPOCHS    = int(os.getenv("EPOCHS", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
SUBCAP    = int(os.getenv("SUBSAMPLE_PER_CLASS", "0"))
PATIENCE  = int(os.getenv("PATIENCE", "2"))

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.0")

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def make_ds(p: Path) -> Dataset:
    rows=[]
    for r in load_jsonl(p):
        if r.get("label") in LABEL2ID and r.get("topic") and r.get("text"):
            rows.append({"query": r["topic"], "text": r["text"], "labels": LABEL2ID[r["label"]]})
    if not rows:
        raise ValueError(f"No usable rows in {p}")
    return Dataset.from_list(rows)

def supported_kwargs(d: dict):
    allowed=set(inspect.signature(TrainingArguments).parameters.keys())
    return {k:v for k,v in d.items() if k in allowed}

def subsample_balanced(ds: Dataset, cap: int) -> Dataset:
    if cap <= 0: return ds
    # Collect indices per class, then keep up to `cap` per label.
    byc = defaultdict(list)
    for i, y in enumerate(ds["labels"]):
        byc[int(y)].append(i)
    keep = []
    rng = random.Random(42)
    for c, idxs in byc.items():
        rng.shuffle(idxs)
        keep.extend(idxs[:cap])
    keep = sorted(keep)
    return ds.select(keep)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=None)  # populated during first forward pass

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
    assert train_path.exists() and dev_path.exists(), "Processed SemEval files not found."

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    def tokenize(batch):
        enc = tok(batch["query"], batch["text"], truncation=True,
                  padding="max_length", max_length=MAX_LEN)
        enc["labels"] = batch["labels"]
        return enc

    ds_tr_raw = make_ds(train_path)
    ds_de_raw = make_ds(dev_path)

    # Optionally downsample each label to shrink dev runs.
    if SUBCAP > 0:
        ds_tr_raw = subsample_balanced(ds_tr_raw, SUBCAP)

    # Class weights encourage macro F1 by offsetting imbalance.
    cnt = Counter(ds_tr_raw["labels"])
    total = sum(cnt.values())
    weights = [ total / (3.0 * max(1, cnt.get(l,1))) for l in [0,1,2] ]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    ds_tr = ds_tr_raw.shuffle(seed=42).map(tokenize, batched=True, remove_columns=["query","text"])
    ds_de = ds_de_raw.map(tokenize, batched=True, remove_columns=["query","text"])

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(os.getenv("LORA_R", "8")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "16")),
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
        target_modules=os.getenv("LORA_TARGETS", "q_lin,k_lin,v_lin,out_lin").split(","),
        bias="none",
    )
    peft_model = get_peft_model(base, lora_cfg)

    # Track which accelerators we can lean on.
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # "epoch"|"steps"|"no" (or "0") per transformers semantics.
    eval_mode = os.getenv("EVAL_MODE", "epoch")
    load_best = eval_mode in {"epoch", "steps"}

    base_args = dict(
        output_dir=str(MODEL_DIR),
        learning_rate=float(os.getenv("LR", "2e-4")),
        per_device_train_batch_size=int(os.getenv("BATCH_TRAIN", "12")),
        per_device_eval_batch_size=int(os.getenv("BATCH_EVAL", "24")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "1")),
        num_train_epochs=EPOCHS,
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
        logging_steps=int(os.getenv("LOG_STEPS", "25")),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=bool(int(os.getenv("GRAD_CHKPT", "0"))),
        optim=os.getenv("OPTIM", "adamw_torch"),
        fp16=(use_cuda and bool(int(os.getenv("FP16", "1")))),
        seed=42,
        no_cuda=not use_cuda and not use_mps,
        save_total_limit=1,
        # transformers renamed this arg across versions; pass both for safety.
        evaluation_strategy=eval_mode,
        eval_strategy=eval_mode,
        save_strategy=eval_mode if eval_mode != "no" else "no",
        load_best_model_at_end=load_best,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
    )
    if MAX_STEPS > 0:
        base_args.update(dict(
            max_steps=MAX_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=min(50, max(1, MAX_STEPS//3)),
            logging_steps=min(25, max(1, MAX_STEPS//6)),
            save_steps=min(50, max(1, MAX_STEPS//3)),
        ))

    args = TrainingArguments(**supported_kwargs(base_args))

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        out = {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1_macro": f1_score(p.label_ids, preds, average="macro"),
        }
        for lab, name in [(0,"favor"), (1,"against"), (2,"none")]:
            try:
                out[f"f1_{name}"] = f1_score(p.label_ids, preds, labels=[lab], average="macro")
            except Exception:
                out[f"f1_{name}"] = 0.0
        return out

    callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]

    trainer = WeightedTrainer(
        model=peft_model,
        args=args,
        tokenizer=tok,
        train_dataset=ds_tr,
        eval_dataset=ds_de,
        compute_metrics=metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    # Apple MPS often prefers fewer intraop threads.
    try:
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
    except Exception:
        pass

    trainer.train()

    # Merge LoRA adapters into the base weights for deployment.
    merged = peft_model.merge_and_unload()
    merged.config.id2label = ID2LABEL
    merged.config.label2id = LABEL2ID

    merged.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    print(f"[OK] LoRA merged model + tokenizer saved to {MODEL_DIR}")
    print(f"[HINT] Set STANCE_MAX_LEN_INFER={MAX_LEN} in your .env to match training.")
if __name__ == "__main__":
    main()
