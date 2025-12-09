#!/usr/bin/env python3
"""
Prepare Reddit JSONL -> data/processed/reddit_clean.jsonl

- Loads all files matching --glob (default: data/raw/kaggle/reddit*.jsonl)
- PII mask (emails, phones, URLs, @handles, long IDs)
- Quality gate (length/density/spam/near-dupe) + NSFW flag + stanceable
- Uses models/stance_cues.json if present (falls back to empty)
- Writes compact JSONL with normalized fields

- Example:
    python -m scripts.prepare_reddit --glob "data/raw/kaggle/reddit*.jsonl" --verbose
"""

import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from glob import glob


# Project paths -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "kaggle"
PRO  = ROOT / "data" / "processed"
PRO.mkdir(parents=True, exist_ok=True)

# Optional fast JSON ------------------------------------------------
try:
    import orjson as _json
    loads = _json.loads
except Exception:
    import json as _json
    loads = _json.loads

# Tiny JSON dumper for consistent UTF-8 output.
def dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)

# Load cue lexicon if present; fall back to empty weights.
CUE_PATH = ROOT / "models" / "stance_cues.json"
try:
    CUES = _json.loads(CUE_PATH.read_bytes())
except Exception:
    CUES = {"favor": {}, "against": {}}

# Ensure stopword list exists before import.
def ensure_nltk_stopwords():
    try:
        from nltk.corpus import stopwords  # noqa
        _ = stopwords.words("english")
    except Exception:
        import nltk
        nltk.download("stopwords")
ensure_nltk_stopwords()
from nltk.corpus import stopwords  # noqa
STOP = set(stopwords.words("english"))

# Quality gate is shared with backend ingestion.
try:
    from backend.utils.quality import quality_gate
except Exception:
    print("ERROR: backend/utils/quality.py not found or import failed.", file=sys.stderr)
    raise

# PII regexes -------------------------------------------------------
RE_EMAIL   = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE   = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
RE_URL     = re.compile(r"(https?://\S+|\bwww\.\S+)")
RE_HANDLE  = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
RE_LONGNUM = re.compile(r"\b\d{12,}\b")

def mask_pii(s: str) -> tuple[str, bool]:
    if not s: return s, False
    masked = s
    before = masked
    masked = RE_EMAIL.sub("[email]", masked)
    masked = RE_PHONE.sub("[phone]", masked)
    masked = RE_URL.sub("[url]", masked)
    masked = RE_HANDLE.sub("[user]", masked)
    masked = RE_LONGNUM.sub("[id]", masked)
    return masked, (masked != before)

from datetime import datetime, timezone
def to_iso8601(epoch_seconds):
    if not epoch_seconds:
        return None
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).isoformat()
    except Exception:
        return None

def main():
    ap = ArgumentParser()
    ap.add_argument("--glob", default=str(RAW / "reddit*.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--simhash-bits", type=int, default=12)   # wired inside quality_gate
    ap.add_argument("--simhash-tol", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(p) for p in glob(args.glob))
    if not files:
        print("No reddit*.jsonl in data/raw/", file=sys.stderr)

    # Lazy import tqdm only when verbose progress is requested.
    tqdm = None
    if args.verbose:
        try:
            from tqdm import tqdm as _tqdm
            tqdm = _tqdm
        except Exception:
            pass

    out_path = PRO / "reddit_clean.jsonl"
    kept = dropped = dups = 0

    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                it = tqdm(f, desc=fp.name, unit="lines") if tqdm else f
                for i, line in enumerate(it, 1):
                    if args.limit and i > args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = loads(line)
                    except Exception:
                        continue

                    rid  = str(r.get("id") or "")
                    body = (r.get("body") or "").strip()
                    subr = r.get("subreddit") or "unknown"
                    ts   = to_iso8601(r.get("created_utc"))
                    score= int(r.get("score") or 0)
                    if not body:
                        continue

                    text, pii = mask_pii(body)

                    # fast quality gate (query-agnostic)
                    qscore, nsfw, stanceable, sh, reason = quality_gate(
                        text, CUES, STOP,
                        seen=None,
                        simhash_tol=(None if args.no_dedupe else args.simhash_tol),
                        bucket_bits=args.simhash_bits,
                    )
                    if reason == "near_dupe":
                        dups += 1
                        continue
                    if qscore < 0.50 or not stanceable:
                        dropped += 1
                        continue

                    doc = {
                        "id": f"reddit-{rid}" if rid else f"reddit-auto",
                        "source": "reddit",
                        "topic": None,
                        "text": text,
                        "label": "none",
                        "timestamp": ts,
                        "nsfw": bool(nsfw),
                        "pii_masked": bool(pii),
                        "quality_score": float(qscore),
                        "stanceable": bool(stanceable),
                        "subreddit": subr,
                        "score": score,
                        "split": "corpus"
                    }
                    out.write(dumps(doc) + "\n")
                    kept += 1

    print(f"reddit_clean.jsonl → {out_path}")
    print(f"Kept: {kept:,} | Dropped: {dropped:,} (near-dupes: {dups:,})")

if __name__ == "__main__":
    main()
