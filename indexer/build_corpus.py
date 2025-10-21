#!/usr/bin/env python3
import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SEMEVAL_DIR = RAW / "semeval"
KAGGLE_REDDIT = RAW / "kaggle" / "reddit_subset.jsonl"

SEMEVAL_OUT = PROCESSED / "semeval_clean.jsonl"
REDDIT_OUT = PROCESSED / "reddit_clean.jsonl"
MERGED_OUT = PROCESSED / "corpus.jsonl"

def write_jsonl(path: Path, it):
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in it:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count

def load_semeval_lines():
    """
    Robust reader for SemEval 2016 Task A "all-annotations" files.
    Handles stray tabs/quotes/BOM and variable columns.
    Expected logical columns: id, target, stance, tweet
    """
    candidates = sorted((SEMEVAL_DIR).glob("*all-annotations*.txt"))
    if not candidates:
        print(f"⚠️  No *all-annotations*.txt files found in {SEMEVAL_DIR}")
        return

    total = 0
    for p in candidates:
        print(f"↪ Parsing {p.name}")
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                row = [c.lstrip("\ufeff") if isinstance(c, str) else c for c in row]
                if row[0].lower() in {"id", "tweet id"}:
                    continue
                if len(row) < 4:
                    continue
                _id, target, stance = row[0], row[1], row[2]
                tweet = row[3] if len(row) == 4 else "\t".join(row[3:])
                if not tweet or tweet.strip().lower() in {"[deleted]", "[removed]"}:
                    continue
                if len(tweet.strip()) < 10:
                    continue
                total += 1
                yield {
                    "id": f"semeval-{_id}",
                    "body": tweet,
                    "source": "semeval",
                    "subreddit": None,
                    "created_utc": None,
                    "score": None,
                    "target": target,
                    "stance_gold": stance,
                }
    if total == 0:
        print("⚠️  Parsed files but found 0 valid SemEval rows. Check file contents.")

def load_reddit_lines():
    if not KAGGLE_REDDIT.exists():
        print(f"⚠️  Missing {KAGGLE_REDDIT}.")
        return
    with KAGGLE_REDDIT.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            body = obj.get("body", "")
            if not body or body.strip().lower() in {"[deleted]", "[removed]"}:
                continue
            if len(body.strip()) < 10:
                continue
            yield {
                "id": f"reddit-{obj.get('id')}",
                "body": body,
                "source": "reddit",
                "subreddit": obj.get("subreddit"),
                "created_utc": obj.get("created_utc"),
                "score": obj.get("score", 0),
                "target": None,
                "stance_gold": None,
            }

def main():
    # SemEval
    semeval_count = write_jsonl(SEMEVAL_OUT, load_semeval_lines() or [])
    print(f"✅ Wrote {semeval_count:,} SemEval rows → {SEMEVAL_OUT}")

    # Reddit
    reddit_count = write_jsonl(REDDIT_OUT, load_reddit_lines() or [])
    print(f"✅ Wrote {reddit_count:,} Reddit rows → {REDDIT_OUT}")

    # Merge (SemEval first)
    merged = 0
    with MERGED_OUT.open("w", encoding="utf-8") as out:
        for p in (SEMEVAL_OUT, REDDIT_OUT):
            if not p.exists(): 
                continue
            with p.open("r", encoding="utf-8") as fin:
                for ln in fin:
                    out.write(ln)
                    merged += 1
    print(f"✅ Merged {merged:,} rows → {MERGED_OUT}")

if __name__ == "__main__":
    main()
