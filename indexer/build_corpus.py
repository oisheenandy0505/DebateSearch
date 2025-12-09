#!/usr/bin/env python3
"""
Builds the unified search corpus:

  - data/processed/semeval_clean.jsonl  (labeled stance dataset)
  - data/processed/reddit_clean.jsonl   (our HF-streamed Reddit subset)

Outputs:
  - data/processed/corpus.jsonl

Each record has:
  id, title, body, source, subreddit, created_utc, score,
  target, stance_gold, quality_score
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]       # DebateSearch/
DATA_DIR = ROOT / "data"                         # DebateSearch/data/

SEM_PATH = DATA_DIR / "processed" / "semeval_clean.jsonl"
RED_PATH = DATA_DIR / "processed" / "reddit_clean.jsonl"
OUT_PATH = DATA_DIR / "processed" / "corpus.jsonl"


def load_jsonl(path: Path):
    if not path.exists():
        print(f" Missing file: {path}")
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sem_rows = load_jsonl(SEM_PATH)
    red_rows = load_jsonl(RED_PATH)

    print(f"Loaded {len(sem_rows):,} SemEval rows")
    print(f"Loaded {len(red_rows):,} Reddit rows")

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        # SemEval first (labeled)
        for i, ex in enumerate(sem_rows, start=1):
            body = ex.get("body") or ex.get("tweet") or ""
            doc = {
                "id": ex.get("id") or f"semeval-{i}",
                "title": ex.get("target") or "(SemEval)",
                "body": body,
                "source": "semeval",
                "subreddit": None,
                "created_utc": None,
                "score": None,
                "target": ex.get("target"),
                "stance_gold": ex.get("stance_gold"),
                "quality_score": 1.0,  # labeled, curated data
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # Reddit (already cleaned + quality_score)
        for ex in red_rows:
            doc = {
                "id": ex.get("id"),
                "title": "(reddit)",
                "body": ex.get("body") or "",
                "source": "reddit",
                "subreddit": ex.get("subreddit"),
                "created_utc": ex.get("created_utc"),
                "score": ex.get("score"),
                "target": ex.get("target"),          # None
                "stance_gold": ex.get("stance_gold"),# None
                "quality_score": ex.get("quality_score", 0.5),
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total = len(sem_rows) + len(red_rows)
    print(f" Wrote {total:,} docs → {OUT_PATH}")


if __name__ == "__main__":
    main()