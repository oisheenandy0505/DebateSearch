#!/usr/bin/env python3
"""
Lightweight quality gate + near-duplicate filtering.

Returns:
  (qscore, nsfw, stanceable, simhash64, reason)

- qscore ∈ [0,1]: higher is better
- nsfw: coarse flag based on lexicon
- stanceable: True if text likely has stance signal
- simhash64: 64-bit int (or None if skipped)
- reason: "ok" | "short" | "symbols" | "near_dupe" | "low_signal"

Near-dup speedup:
- Bucketed SimHash: compare only with same 12-bit prefix (≈4096 buckets)
- Bound bucket size to avoid O(N²)
"""

from __future__ import annotations
import re
import math
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple, Optional

# ---------------- SimHash helpers ----------------

def _tokens_for_simhash(text: str):
    # simple 3-gram shingle of alnum tokens
    words = re.findall(r"[a-z0-9]+", text.lower())
    for i in range(len(words)):
        yield words[i]
        if i+1 < len(words):
            yield words[i] + " " + words[i+1]
        if i+2 < len(words):
            yield words[i] + " " + words[i+1] + " " + words[i+2]

def simhash64(text: str) -> int:
    bits = [0] * 64
    for tok in _tokens_for_simhash(text):
        h = hash(tok) & ((1 << 64) - 1)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, v in enumerate(bits):
        if v >= 0:
            out |= (1 << i)
    return out

# ---------- Fast near-dupe via buckets ----------

_SIMHASH_BUCKETS = defaultdict(list)

def _bucket_key(sh: int, bucket_bits: int = 12) -> int:
    return sh >> (64 - bucket_bits)

def near_dupe(sh: int, simhash_tol: int = 4, bucket_bits: int = 12, max_per_bucket: int = 500) -> bool:
    """Return True if sh is near-duplicate of something already seen."""
    b = _bucket_key(sh, bucket_bits)
    cand = _SIMHASH_BUCKETS.get(b, [])
    # bounded comparisons within bucket
    for prev in cand:
        if (sh ^ prev).bit_count() <= simhash_tol:
            return True
    # learn
    lst = _SIMHASH_BUCKETS[b]
    if len(lst) < max_per_bucket:
        lst.append(sh)
    else:
        lst.pop(0); lst.append(sh)
    return False

# ---------------- Heuristics & lexica ----------------

RE_ALNUM   = re.compile(r"[A-Za-z0-9]")
RE_WORD    = re.compile(r"[A-Za-z']+")
RE_NSFW    = re.compile(r"\b(fuck|shit|bitch|porn|nsfw|bastard|slut|dick|cunt)\b", re.I)

def _signal_score(text: str, cues: Dict[str, Dict[str, float]], stop: Set[str]) -> float:
    """Crude stance-signal score from cue hits + content density."""
    low = text.lower()
    cue_f = sum(w for k, w in cues.get("favor", {}).items() if k in low)
    cue_a = sum(w for k, w in cues.get("against", {}).items() if k in low)
    cue = max(cue_f, cue_a)
    words = [w for w in RE_WORD.findall(low)]
    if not words:
        return 0.0
    non_stop = sum(1 for w in words if w not in stop)
    density = non_stop / max(5, len(words))    # [0..1]
    # bounded blend
    return max(0.0, min(1.0, 0.5 * density + 0.5 * (1 if cue > 0 else 0)))

def quality_gate(
    text: str,
    cues: Dict[str, Dict[str, float]],
    stop: Set[str],
    seen: Optional[Set[int]] = None,             # not used by fast near_dupe; kept for API compat
    simhash_tol: Optional[int] = 4,
    bucket_bits: int = 12,
) -> Tuple[float, bool, bool, Optional[int], str]:

    # fast rejects
    t = text.strip()
    if len(t) < 40:
        return (0.2, False, False, None, "short")

    alnum = sum(1 for ch in t if RE_ALNUM.match(ch))
    ratio = alnum / max(1, len(t))
    if ratio < 0.50:  # too many symbols/links/junk
        return (0.3, False, False, None, "symbols")

    # nsfw flag (coarse)
    nsfw = bool(RE_NSFW.search(t))

    # stance signal
    sig = _signal_score(t, cues, stop)
    stanceable = sig >= 0.25
    base_q = 0.4 + 0.6 * sig         # scale into [0.4..1.0]
    qscore = max(0.0, min(1.0, base_q - (0.1 if nsfw else 0.0)))

    # skip simhash for low-signal/very short to save time
    sh = None
    if simhash_tol is not None and len(t) >= 60 and stanceable:
        sh = simhash64(t)
        if near_dupe(sh, simhash_tol=simhash_tol, bucket_bits=bucket_bits):
            return (qscore, nsfw, stanceable, sh, "near_dupe")

    if not stanceable:
        return (qscore, nsfw, False, sh, "low_signal")

    return (qscore, nsfw, True, sh, "ok")