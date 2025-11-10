#!/usr/bin/env python3
"""
Ad-hoc sanity check for stance clustering with cue-based tie-break.
"""
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cues import tie_break_label

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "stance_distilbert"
PRO = ROOT / "data" / "processed"

LABELS = ["favor","against","none"]

def softmax(x): 
    x = torch.tensor(x, dtype=torch.float32)
    x = x - x.max()
    e = torch.exp(x)
    return (e / e.sum()).tolist()

def load_docs(n=30):
    # quick sample from reddit + semeval clean sets
    paths = [PRO/"reddit_clean.jsonl", PRO/"semeval_test.jsonl", PRO/"semeval_dev.jsonl"]
    docs = []
    for p in paths:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try: obj = json.loads(line)
                    except: continue
                    txt = obj.get("text") or obj.get("body")
                    if not txt: continue
                    docs.append(txt)
                    if len(docs) >= n: return docs
    return docs

def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    query = input("Query (e.g., 'cryptocurrency regulation'): ").strip()
    texts = load_docs(60)

    buckets = {0:[], 1:[], 2:[]}
    with torch.no_grad():
        for t in texts:
            enc = tok(query, t, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
            out = model(**enc)
            probs = softmax(out.logits[0].tolist())
            pred = int(torch.argmax(out.logits, dim=1).item())
            # apply cue tie-break
            pred = tie_break_label(query, t, probs)
            buckets[pred].append((max(probs), t))

    print("\nCounts:", {LABELS[k]: len(v) for k,v in buckets.items()})
    for k in (0,1,2):
        print(f"\n=== {LABELS[k].upper()} (top 3) ===")
        for p, t in sorted(buckets[k], reverse=True)[:3]:
            print(f"- {p:.2f} :: {t[:160].replace('\\n',' ')}")

if __name__ == "__main__":
    main()