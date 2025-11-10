#!/usr/bin/env python3
"""
Combine SemEval and Reddit into one corpus for retrieval (labels kept if present).
Output:
  data/processed/corpus.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRO = ROOT / "data" / "processed"
OUT = PRO / "corpus.jsonl"

def emit(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    files = [
        PRO / "semeval_train.jsonl",
        PRO / "semeval_dev.jsonl",
        PRO / "semeval_test.jsonl",
        PRO / "reddit_clean.jsonl",
    ]
    files = [p for p in files if p.exists()]
    n = 0
    with OUT.open("w", encoding="utf-8") as w:
        for fp in files:
            for obj in emit(fp):
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} records -> {OUT}")

if __name__ == "__main__":
    main()
