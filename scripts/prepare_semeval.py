#!/usr/bin/env python3
"""
Prepare SemEval-2016 Task 6 (stance) -> data/processed/semeval_{train,dev,test}.jsonl.

Accepts any tab-separated file in data/raw/semeval/ matching common names:
  trainingdata-*.txt/.tsv, trialdata-*.txt/.tsv, testdata-*.txt/.tsv, etc.

- PII masking
- Quality gate (lenient; still drops near-dupes & very low signal)
- Stratified split if no official dev/test provided

Example:
    python -m scripts.prepare_semeval --default
"""

import csv
import re
import sys
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "semeval"
PRO  = ROOT / "data" / "processed"
PRO.mkdir(parents=True, exist_ok=True)

# Optional fast JSON ------------------------------------------------
try:
    import orjson as _json
    dumps = lambda o: _json.dumps(o).decode("utf-8")
except Exception:
    import json as _json
    dumps = lambda o: _json.dumps(o, ensure_ascii=False)

# Cue lexicon mirrors the Reddit cleaner; fall back to empty weights.
CUE_PATH = ROOT / "models" / "stance_cues.json"
try:
    CUES = _json.loads(CUE_PATH.read_bytes())
except Exception:
    CUES = {"favor": {}, "against": {}}

# Ensure stopwords exist locally for nltk.
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

# Shared quality gate keeps thresholds aligned with the backend.
try:
    from backend.utils.quality import quality_gate
except Exception:
    print("ERROR: backend/utils/quality.py not found or import failed.", file=sys.stderr)
    raise

# PII regexes (mirrors prepare_reddit).
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

def find_semeval_tsvs():
    patterns = [
        "trainingdata*.tsv", "trainingdata*.txt",
        "trialdata*.tsv",    "trialdata*.txt",
        "testdata*.tsv",     "testdata*.txt",
        "*task*A*train*.tsv","*task*A*train*.txt",
        "*.tsv", "*.txt",
    ]
    seen = set()
    for pat in patterns:
        hits = [fp for fp in sorted(RAW.glob(pat)) if fp.suffix.lower() in {".tsv", ".txt"}]
        for fp in hits:
            if fp not in seen:
                seen.add(fp)
                yield fp
        if seen:
            return

def normalize_row(row: dict) -> dict | None:
    lower = {k.lower().strip(): v for k, v in row.items()}
    id_     = lower.get("id") or lower.get("tweetid") or lower.get("tweet id") or ""
    target  = lower.get("target") or lower.get("topic") or ""
    tweet   = lower.get("tweet") or lower.get("text") or ""
    stance  = lower.get("stance") or lower.get("label") or ""

    if not tweet or not target:
        return None

    s = stance.strip().lower()
    if s in ("favor", "against", "none"):
        label = s
    elif s in ("positive", "pos", "for", "support", "pro"):
        label = "favor"
    elif s in ("negative", "neg", "against", "anti", "con"):
        label = "against"
    else:
        label = "none"

    return {
        "id": str(id_).strip() or None,
        "topic": target.strip(),
        "text": tweet.strip(),
        "label": label
    }

def stratified_split(rows, seed=42, train_ratio=0.8, dev_ratio=0.1):
    random.seed(seed)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["topic"], r["label"])].append(r)

    train, dev, test = [], [], []
    for _, grp in by_key.items():
        random.shuffle(grp)
        n = len(grp)
        n_tr = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train += grp[:n_tr]
        dev   += grp[n_tr:n_tr+n_dev]
        test  += grp[n_tr+n_dev:]
    return train, dev, test

def main():
    files = list(find_semeval_tsvs())
    if not files:
        print("No semeval TSV/TXT files found in data/raw/semeval/", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            # Detect header: try to see if typical columns present
            has_header = False
            if header:
                hdr_low = [h.strip().lower() for h in header]
                if any(h in hdr_low for h in ("id", "tweet", "stance", "target")):
                    has_header = True
            if not has_header:
                # treat first line as data
                f.seek(0)
                reader = csv.reader(f, delimiter="\t")
            else:
                # keep header for named mapping
                pass

            if has_header:
                hdr = [h.strip() for h in header]
                for cols in reader:
                    if len(cols) != len(hdr):
                        continue
                    row = dict(zip(hdr, cols))
                    norm = normalize_row(row)
                    if norm: rows.append(norm)
            else:
                for cols in reader:
                    if len(cols) < 4:
                        continue
                    row = {
                        "ID": cols[0],
                        "Target": cols[1],
                        "Tweet": cols[2],
                        "Stance": cols[3],
                    }
                    norm = normalize_row(row)
                    if norm: rows.append(norm)

    if not rows:
        print("No usable SemEval rows parsed.", file=sys.stderr)
        sys.exit(2)

    # Clean & score (lenient thresholds keep most tweets).
    cleaned = []
    kept = dropped = dups = 0
    for r in rows:
        txt, pii = mask_pii(r["text"])
        qscore, nsfw, stanceable, sh, reason = quality_gate(
            txt, CUES, STOP,
            seen=None,            # fast near-dupe buckets are internal
            simhash_tol=4,        # gentle dedupe
            bucket_bits=12,
        )
        if reason == "near_dupe":
            dups += 1
            continue
        if qscore < 0.35:
            dropped += 1
            continue

        cleaned.append({
            "id": f"semeval-{r['id']}" if r.get("id") else None,
            "source": "semeval",
            "topic": r["topic"],
            "text": txt,
            "label": r["label"],
            "timestamp": None,
            "nsfw": bool(nsfw),
            "pii_masked": bool(pii),
            "quality_score": float(qscore),
            "stanceable": True,
        })

    train, dev, test = stratified_split(cleaned, seed=42, train_ratio=0.8, dev_ratio=0.1)

    def dump(path: Path, items, split_name: str):
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(items, 1):
                obj = dict(r)
                obj["split"] = split_name
                if not obj.get("id"):
                    obj["id"] = f"semeval-{split_name}-{i}"
                f.write(dumps(obj) + "\n")

    dump(PRO / "semeval_train.jsonl", train, "train")
    dump(PRO / "semeval_dev.jsonl",   dev,   "dev")
    dump(PRO / "semeval_test.jsonl",  test,  "test")

    print(f"Wrote: {len(train):,} train, {len(dev):,} dev, {len(test):,} test")
    print(f"Dropped: {dropped:,} (near-dupes: {dups:,})")

if __name__ == "__main__":
    main()
