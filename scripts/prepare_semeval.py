#!/usr/bin/env python3
"""
Convert SemEval 2016 Task 6 raw files into JSONL splits compatible with stance training.

Inputs (expected):
data/raw/semeval/
  ├─ trainingdata-all-annotations.txt
  ├─ trainingdata-ids.txt
  ├─ trialdata-all-annotations.txt
  ├─ trialdata-ids.txt
  ├─ testdata-taskA-all-annotations.txt
  ├─ testdata-taskA-ids.txt
  ├─ readme.txt (ignored)

Outputs:
data/processed/
  ├─ semeval_train.jsonl
  ├─ semeval_dev.jsonl
  └─ semeval_test.jsonl

Each JSON line:
{"id": "...", "source": "semeval", "topic": "...", "text": "...", "label": "favor|against|none", "split": "..."}
"""
import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
RAW = ROOT / "data" / "raw" / "semeval"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# SemEval Task 6 A file format: tab-separated columns with tweet id, target, sentiment label, and text.
# Lines look like:
# 123456789012345678	Atheism	AGAINST	"Tweet text ..."
# or sometimes quoted text without inner tabs.

def parse_tsv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("tweetid"):
                continue
            # Split only first 3 tabs, remaining is text
            parts = line.split("\t")
            if len(parts) < 4:
                # some files may use 3 parts with text containing tabs; best-effort join
                # try to reconstruct: id, target, stance, text
                if len(parts) >= 3:
                    pid, target, stance = parts[0], parts[1], parts[2]
                    text = "" if len(parts) == 3 else "\t".join(parts[3:])
                else:
                    continue
            else:
                pid, target, stance = parts[0], parts[1], parts[2]
                text = "\t".join(parts[3:])
            text = text.strip().strip('"')
            label = stance.lower()
            if label == "favor":  lab = "favor"
            elif label == "against": lab = "against"
            else: lab = "none"
            rows.append({
                "id": f"semeval-{pid}",
                "source": "semeval",
                "topic": target.strip(),
                "text": text,
                "label": lab
            })
    return rows

def write_jsonl(path: Path, rows, split: str):
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            r2 = dict(r)
            r2["split"] = split
            w.write(json.dumps(r2, ensure_ascii=False) + "\n")

def main():
    train_rows = parse_tsv(RAW / "trainingdata-all-annotations.txt")
    dev_rows   = parse_tsv(RAW / "trialdata-all-annotations.txt")
    test_rows  = parse_tsv(RAW / "testdata-taskA-all-annotations.txt")

    if not train_rows:
        raise SystemExit("No training rows found. Check data/raw/semeval/*.txt paths.")
    if not dev_rows:
        # If no official dev, carve 10% from train
        cut = max(1, int(0.1 * len(train_rows)))
        dev_rows, train_rows = train_rows[:cut], train_rows[cut:]

    write_jsonl(OUT / "semeval_train.jsonl", train_rows, "train")
    write_jsonl(OUT / "semeval_dev.jsonl",   dev_rows,   "dev")
    if test_rows:
        write_jsonl(OUT / "semeval_test.jsonl",  test_rows,  "test")

    print(f"Wrote: {OUT/'semeval_train.jsonl'}, {OUT/'semeval_dev.jsonl'}"
          + (f", {OUT/'semeval_test.jsonl'}" if (OUT/'semeval_test.jsonl').exists() else ""))

if __name__ == "__main__":
    main()
