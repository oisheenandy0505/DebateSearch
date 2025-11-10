#!/usr/bin/env python3
"""
Build stance cue lexicons (global and per-topic) using chi-square:

Output JSON at models/stance_cues.json:
{
  "global": { "favor": {...}, "against": {...} },
  "per_topic": {
      "Atheism": { "favor": {...}, "against": {...} },
      "Legalization of Abortion": { ... },
      ...
  },
  "meta": { "min_df": 10, "top_k": 400 }
}
"""
import json, math, re, collections
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
PRO   = ROOT / "data" / "processed"
OUT   = ROOT / "models" / "stance_cues.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
STOP = {
    "the","a","an","and","or","but","for","nor","so","to","of","in","on","at",
    "is","are","was","were","be","been","am","it","that","this","these","those",
    "i","you","he","she","we","they","me","him","her","us","them","my","your",
    "our","their","with","as","by","from","about","not","no","do","does","did",
}
LABELS = ("favor","against","none")
USE_LABELS = ("favor","against")  # cues only for polar classes

def tokens(s: str):
    for t in TOKEN_RE.findall(s.lower()):
        if t not in STOP and len(t) >= 2:
            yield t

def ngrams(words, n=2):
    # unigrams + bigrams
    ws = list(words)
    for w in ws: yield w
    for i in range(len(ws)-1):
        yield ws[i] + " " + ws[i+1]

def load_split(name):
    path = PRO / f"semeval_{name}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def chi_square(k11, k12, k21, k22):
    # classic 2x2 chi2 with Yates continuity correction
    N = k11 + k12 + k21 + k22
    if N == 0: return 0.0
    num = N * (abs(k11*k22 - k12*k21) - N/2)**2
    den = (k11+k12)*(k21+k22)*(k11+k21)*(k12+k22)
    if den <= 0: return 0.0
    return num / den

def build(min_df=10, top_k=400):
    train = load_split("train")
    # Global counters
    df_term = collections.Counter()                   # document frequency across all docs
    df_label = {lbl: collections.Counter() for lbl in USE_LABELS}
    # Per-topic counters
    per_topic_df = collections.defaultdict(lambda: collections.Counter())
    per_topic_df_label = collections.defaultdict(lambda: {lbl: collections.Counter() for lbl in USE_LABELS})

    all_docs = 0
    by_label_docs = collections.Counter()
    per_topic_docs = collections.defaultdict(int)
    per_topic_bylabel_docs = collections.defaultdict(lambda: collections.Counter())

    for r in train:
        text = r.get("text","")
        topic = r.get("topic","").strip()
        label = r.get("label")
        if not text or label not in LABELS: continue
        all_docs += 1
        if label in USE_LABELS:
            by_label_docs[label] += 1
        if topic:
            per_topic_docs[topic] += 1
            if label in USE_LABELS:
                per_topic_bylabel_docs[topic][label] += 1

        terms = set(ngrams(tokens(text)))
        for t in terms:
            df_term[t] += 1
            if label in USE_LABELS:
                df_label[label][t] += 1
            if topic:
                per_topic_df[topic][t] += 1
                if label in USE_LABELS:
                    per_topic_df_label[topic][label][t] += 1

    # Compute chi2 globally
    global_cues = {lbl:{} for lbl in USE_LABELS}
    for t, df in df_term.items():
        if df < min_df: continue
        for lbl in USE_LABELS:
            k11 = df_label[lbl][t]                         # term & label
            k12 = df_term[t] - k11                         # term & not label
            k21 = by_label_docs[lbl] - k11                # label & not term
            k22 = (all_docs - by_label_docs[lbl]) - k12   # neither
            score = chi_square(k11,k12,k21,k22)
            if score > 0:
                global_cues[lbl][t] = score

    # Per-topic chi2
    per_topic_cues = {}
    for topic, tdf in per_topic_df.items():
        topic_total = per_topic_docs[topic]
        if topic_total < 25:   # tiny topics = noisy
            continue
        cues = {lbl:{} for lbl in USE_LABELS}
        for t, df in tdf.items():
            if df < max(5, min_df//2):  # slightly looser inside topic
                continue
            for lbl in USE_LABELS:
                k11 = per_topic_df_label[topic][lbl][t]
                k12 = df - k11
                k21 = per_topic_bylabel_docs[topic][lbl] - k11
                k22 = (topic_total - per_topic_bylabel_docs[topic][lbl]) - k12
                score = chi_square(k11,k12,k21,k22)
                if score > 0:
                    cues[lbl][t] = score
        # keep top_k per label
        per_topic_cues[topic] = {
            "favor": dict(sorted(cues["favor"].items(), key=lambda x: -x[1])[:top_k]),
            "against": dict(sorted(cues["against"].items(), key=lambda x: -x[1])[:top_k]),
        }

    # trim global too
    global_cues = {
        "favor":  dict(sorted(global_cues["favor"].items(),   key=lambda x: -x[1])[:top_k]),
        "against":dict(sorted(global_cues["against"].items(), key=lambda x: -x[1])[:top_k]),
    }

    out = {
        "global": global_cues,
        "per_topic": per_topic_cues,
        "meta": {"min_df": min_df, "top_k": top_k, "docs": all_docs}
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"saved stance cues → {OUT} "
          f"({len(global_cues['favor'])} favor / {len(global_cues['against'])} against, "
          f"{len(per_topic_cues)} topics)")

if __name__ == "__main__":
    build()