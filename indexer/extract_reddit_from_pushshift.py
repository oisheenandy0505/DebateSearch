#!/usr/bin/env python3
"""
Stream RC_YYYY-MM.bz2 from Pushshift-style dumps and extract *only* high-signal
political / debate comments.

Filters:
- subreddit must be in ALLOWED_SUBREDDITS
- body not [deleted] / [removed]
- score >= MIN_SCORE
- len(body) between MIN_BODY_LEN and MAX_BODY_LEN
- "English-ish": enough letters + some common words
- "Debate-ish": should/because/argue/etc., min token count
- NSFW-ish comments are dropped (simple keyword filter)
- PII (emails, URLs, phones, @handles, u/handles) is masked

Output: line-delimited JSON to a .jsonl file, capped at --max-comments.
"""

import argparse
import bz2
import json
import re
from pathlib import Path
from typing import Tuple

ALLOWED_SUBREDDITS = {
    "politics",
    "worldpolitics",
    "PoliticalDiscussion",
    "AskPolitics",
    "changemyview",
    "neutralpolitics",
}

MIN_SCORE = 3
MIN_BODY_LEN = 120
MAX_BODY_LEN = 2000
MIN_ALPHA_RATIO = 0.6
MIN_ENGLISH_WORDS = 3
MIN_TOKENS = 15

COMMON_EN_WORDS = {
    "the", "and", "that", "this", "you", "have", "with",
    "for", "from", "not", "are", "will", "they", "about",
}

DEBATE_WORDS = {
    "should", "must", "because", "argue", "argument",
    "evidence", "believe", "think", "support", "oppose",
    "against", "policy", "law", "rights", "ban", "legal",
}

# PII patterns
PII_PATTERNS = [
    (re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.I), "[EMAIL]"),
    (re.compile(r'https?://\S+', re.I), "[URL]"),
    (re.compile(r'\b(?:www\.)\S+', re.I), "[URL]"),
    (re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[PHONE]"),
    (re.compile(r'\b@[A-Za-z0-9_]+\b'), "[HANDLE]"),
    (re.compile(r'\bu/[A-Za-z0-9_-]+\b'), "[USER]"),
]

# Simple NSFW-ish keywords; intentionally generic (no slurs / explicit acts)
NSFW_KEYWORDS = {
    "porn", "nsfw", "xxx", "fetish",
    "beheading", "decapitation", "gore",
}


def clean_body(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_english(text: str) -> bool:
    if not text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    ratio = letters / max(1, len(text))
    if ratio < MIN_ALPHA_RATIO:
        return False

    lower = text.lower()
    hits = sum(1 for w in COMMON_EN_WORDS if w in lower)
    return hits >= MIN_ENGLISH_WORDS


def is_debate_like(text: str) -> bool:
    lower = text.lower()
    tokens = lower.split()
    if len(tokens) < MIN_TOKENS:
        return False
    hits = sum(1 for w in DEBATE_WORDS if w in lower)
    return hits >= 1


def mask_pii(text: str) -> str:
    out = text
    for patt, repl in PII_PATTERNS:
        out = patt.sub(repl, out)
    return out


def is_nsfw(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in NSFW_KEYWORDS)


def keep_comment(obj: dict) -> Tuple[bool, str]:
    # Pull the pieces we inspect and normalize the body once upfront.
    body = obj.get("body") or ""
    subreddit = obj.get("subreddit") or ""
    score = obj.get("score") or 0

    body = clean_body(body)

    # Drop obvious junk
    if not body:
        return False, ""
    if body.lower() in {"[deleted]", "[removed]"}:
        return False, ""
    if subreddit not in ALLOWED_SUBREDDITS:
        return False, ""
    if score < MIN_SCORE:
        return False, ""
    if len(body) < MIN_BODY_LEN or len(body) > MAX_BODY_LEN:
        return False, ""
    if not looks_english(body):
        return False, ""
    if not is_debate_like(body):
        return False, ""
    if is_nsfw(body):
        return False, ""

    body = mask_pii(body)
    return True, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to RC_YYYY-MM.bz2")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--max-comments", type=int, default=1_000_000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0

    with bz2.open(in_path, "rt", encoding="utf-8", errors="ignore") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ok, body = keep_comment(obj)
            if not ok:
                continue

            out = {
                "id": obj.get("id") or obj.get("name") or f"reddit-{seen}",
                "subreddit": obj.get("subreddit"),
                "body": body,
                "created_utc": obj.get("created_utc"),
                "score": obj.get("score", 0),
                "source": "reddit",
                "target": None,
                "stance_gold": None,
            }
            fout.write(json.dumps(out) + "\n")
            kept += 1

            if kept % 10_000 == 0:
                print(f"[INFO] kept {kept} / seen {seen}")
            if kept >= args.max_comments:
                print(f"[DONE] Hit cap: {kept} comments.")
                break

    print(f"[SUMMARY] seen={seen}, kept={kept}, output={out_path}")


if __name__ == "__main__":
    main()
