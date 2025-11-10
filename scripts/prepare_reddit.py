#!/usr/bin/env python3
"""
Clean Reddit subset into JSONL for retrieval (unlabeled).

Input candidates (first existing wins):
- data/raw/kaggle/reddit_subset.jsonl
- data/raw/kaggle/reddit_subset.json
- data/raw/kaggle/reddit_subset.json.gz

Output:
- data/processed/reddit_clean.jsonl

JSON line fields:
{"id":"reddit-<orig_id|line#>","source":"reddit","topic":null,"text":"...", "label":null, "meta":{...}}
"""
import json, gzip
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kaggle"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def iter_any():
    paths = [
        RAW / "reddit_subset.jsonl",
        RAW / "reddit_subset.json",
        RAW / "reddit_subset.json.gz",
    ]
    path = next((p for p in paths if p.exists()), None)
    if not path:
        raise SystemExit("No reddit subset found in data/raw/kaggle/")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                yield obj

def clean_text(t: str) -> str:
    t = t or ""
    # remove markdown links, URLs, and excessive whitespace
    t = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    out = (OUT / "reddit_clean.jsonl").open("w", encoding="utf-8")
    n_in = n_out = 0
    for n_in, row in enumerate(iter_any(), start=1):
        text = clean_text(row.get("body") or row.get("text") or row.get("content") or "")
        if not text or text == "[deleted]" or text == "[removed]":
            continue
        rid = row.get("id") or f"line{n_in}"
        meta = {
            "subreddit": row.get("subreddit"),
            "created_utc": row.get("created_utc"),
            "author": row.get("author"),
            "score": row.get("score"),
        }
        obj = {
            "id": f"reddit-{rid}",
            "source": "reddit",
            "topic": None,
            "text": text,
            "label": None,
            "meta": meta
        }
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n_out += 1
    out.close()
    print(f"Kept {n_out}/{n_in} reddit rows -> {OUT/'reddit_clean.jsonl'}")

if __name__ == "__main__":
    main()
