"""
Cue utilities: choose topic, score text, and tie-break 'none' using cues.

Usage:
from backend.ml.cues import load_cues, tie_break_label

label = tie_break_label(query, text, probs)  # returns 0=favor,1=against,2=none
"""
from pathlib import Path
import json, math, re
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
CUES_PATH = ROOT / "models" / "stance_cues.json"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
def tok(s): return [t.lower() for t in TOKEN_RE.findall(s or "") if len(t) >= 2]

def unigrams_bigrams(words):
    ws = list(words)
    yield from ws
    for i in range(len(ws)-1):
        yield ws[i] + " " + ws[i+1]

def load_cues():
    data = json.loads(CUES_PATH.read_text(encoding="utf-8"))
    return data

def best_topic(query, topics, threshold=0.15):
    q = set(tok(query))
    if not q: return None
    best, best_j = None, 0.0
    for t in topics:
        tt = set(tok(t))
        if not tt: continue
        inter = len(q & tt)
        uni   = len(q | tt)
        j = inter / uni if uni else 0.0
        if j > best_j:
            best, best_j = t, j
    return best if best_j >= threshold else None

def cue_score(text_terms, cue_dict, query_terms, query_boost=1.5):
    score = 0.0
    # query filter: boost cues that overlap with query terms
    qset = set(query_terms)
    for term, w in cue_dict.items():
        if term in text_terms:
            s = w
            # small boost if cue term shares any token with query
            if qset and any(t in qset for t in term.split()):
                s *= query_boost
            score += s
    return score

def tie_break_label(query, text, probs, none_index=2, delta=0.10):
    """
    probs: iterable of length 3 (favor, against, none).
    If top is 'none' but margin <= delta, use cues to force decision.
    """
    cues = load_cues()
    labels = ["favor","against","none"]

    # quick top2
    ranked = sorted([(p,i) for i,p in enumerate(probs)], reverse=True)
    top_p, top_i = ranked[0]
    second_p, second_i = ranked[1]

    if top_i != none_index:
        return top_i

    if (top_p - second_p) > delta:
        return top_i  # model is confident it's none

    # select per-topic cues first
    topic = best_topic(query, list(cues.get("per_topic", {}).keys()))
    favor_c, against_c = None, None
    if topic and topic in cues["per_topic"]:
        favor_c  = cues["per_topic"][topic]["favor"]
        against_c= cues["per_topic"][topic]["against"]

    # fallback to global if per-topic sparse
    if not favor_c or not against_c:
        favor_c  = cues["global"]["favor"]
        against_c= cues["global"]["against"]

    q_terms  = tok(query)
    t_terms  = list(unigrams_bigrams(tok(text)))

    favor_score   = cue_score(t_terms, favor_c,   q_terms)
    against_score = cue_score(t_terms, against_c, q_terms)

    if favor_score <= 0 and against_score <= 0:
        return top_i  # still none

    return 0 if favor_score >= against_score else 1
