#!/usr/bin/env python3
"""
Stream Reddit comments from Hugging Face (fddemarco/pushshift-reddit-comments),
apply VERY strict quality filters, and write a ~1M-document JSONL file.

Output schema matches the rest of the pipeline:
  id, body, subreddit, created_utc, score, source, target, stance_gold, quality_score
"""

import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]         # DebateSearch/
DATA_DIR = ROOT / "data"                           # DebateSearch/data/
OUT_PATH = DATA_DIR / "processed" / "reddit_clean.jsonl"

# How many *kept* docs you want
TARGET_DOCS = 1_000_000

# Hard cap on how many rows we even look at (saves your laptop)
MAX_STREAM_ROWS = 20_000_000

# Debate-ish subreddits only (very strict)
DEBATE_SUBS = {
    "politics",
    "worldnews",
    "news",
    "neutralpolitics",
    "changemyview",
    "cmv",
    "politicaldiscussion",
    "askpolitics",
    "askphilosophy",
    "philosophy",
    "economics",
    "climate",
    "climateskeptics",
    "ukpolitics",
    "canadapolitics",
    "europe",
}

URL_RE = re.compile(r"https?://\S+")
JUNK_PHRASES = {
    "lol", "lmao", "rofl",
    "this.", "based", "cringe",
    "first!", "ok", "k", "idk",
}


def is_junk_body(text: str) -> bool:
    """Crazy-strict junk filter for Reddit comments."""
    if text is None:
        return True

    body = text.strip()
    if not body:
        return True

    low = body.lower()

    # Deleted/removed markers
    if low in {"[deleted]", "[removed]"}:
        return True

    # Very short / one-liners
    if len(body) < 40:
        return True
    if len(body.split()) < 6:
        return True

    # Obvious junk phrases (exact match)
    if low in JUNK_PHRASES:
        return True

    # Link-only or mostly links
    if body.startswith("http://") or body.startswith("https://"):
        return True
    if URL_RE.search(body) and len(body) < 80:
        return True

    # Excessive caps “rage”
    letters = [c for c in body if c.isalpha()]
    if letters:
        num_caps = sum(c.isupper() for c in letters)
        frac_caps = num_caps / len(letters)
        if num_caps >= 10 and frac_caps > 0.6:
            return True

    # Mostly non-letter crap
    num_letters = sum(c.isalpha() for c in body)
    if num_letters / max(len(body), 1) < 0.4:
        return True

    # Stupidly long (API abuse / copy-paste walls)
    if len(body) > 3000:
        return True

    # Very heavy quoting: starts with ">" and not much original content
    if body.lstrip().startswith(">") and len(body) < 200:
        return True

    return False


def compute_quality_score(body: str, score: int | float | None) -> float:
    """
    Heuristic 0–1 quality score:
      - favor longer, more substantive comments
      - favor higher Reddit score
      - penalize short or URL-heavy comments
    """
    if body is None:
        return 0.0

    body = body.strip()
    length = len(body)
    base = 0.2

    # length_score: 0 at 0 chars → 1 at 400+ chars
    length_score = min(length / 400.0, 1.0)

    # reddit score (clip to [0, 50]) → 0–1
    score = score or 0
    try:
        score = float(score)
    except Exception:
        score = 0.0
    score = max(score, 0.0)
    score_norm = min(score, 50.0) / 50.0

    # penalties
    short_penalty = 0.0
    if length < 120:
        short_penalty = 0.25
    elif length < 200:
        short_penalty = 0.1

    url_penalty = 0.15 if URL_RE.search(body) else 0.0

    q = base + 0.5 * length_score + 0.3 * score_norm - short_penalty - url_penalty
    q = max(0.0, min(1.0, q))
    return round(q, 3)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("→ Streaming from Hugging Face: fddemarco/pushshift-reddit-comments")
    print(f"  Target docs: {TARGET_DOCS:,}")
    print(f"  Max stream rows: {MAX_STREAM_ROWS:,}")

    ds_iter = load_dataset(
        "fddemarco/pushshift-reddit-comments",
        split="train",
        streaming=True,
    )

    n_written = 0
    n_seen = 0

    with OUT_PATH.open("w", encoding="utf-8") as fout, tqdm(
        total=MAX_STREAM_ROWS, desc="Streaming & filtering", unit="rows"
    ) as pbar:
        for ex in ds_iter:
            n_seen += 1
            if n_seen > MAX_STREAM_ROWS:
                break

            body = ex.get("body") or ""
            subreddit = (ex.get("subreddit") or "").lower()

            # subreddit filter
            if subreddit not in DEBATE_SUBS:
                pbar.update(1)
                continue

            # crazy-strict junk filter
            if is_junk_body(body):
                pbar.update(1)
                continue

            # score filter
            score = ex.get("score") or 0
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            if score_f < 3:  # throw away low-engagement
                pbar.update(1)
                continue

            # finalize + quality score
            q_score = compute_quality_score(body, score_f)

            doc = {
                "id": f"reddit-{ex.get('id')}",
                "body": body,
                "subreddit": subreddit,
                "created_utc": ex.get("created_utc"),
                "score": score_f,
                "source": "reddit",
                "target": None,
                "stance_gold": None,
                "quality_score": q_score,
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_written += 1
            pbar.update(1)

            if n_written % 10_000 == 0:
                print(f"  → kept {n_written:,} (seen {n_seen:,})")

            if n_written >= TARGET_DOCS:
                break

    print(f"\nDone.")
    print(f"  Seen rows:   {n_seen:,}")
    print(f"  Kept rows:   {n_written:,}")
    print(f"  Output file: {OUT_PATH}")


if __name__ == "__main__":
    main()