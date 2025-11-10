#!/usr/bin/env python3
import os, json
from pathlib import Path
from collections import Counter
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

OS_URL   = os.getenv("OS_URL", "http://localhost:9200")
OS_INDEX = os.getenv("OS_INDEX", "debate_docs")
MODEL_DIR = os.getenv("STANCE_MODEL_DIR", "models/stance_distilbert")
ID2LABEL = {0:"favor", 1:"against", 2:"none"}

def retrieve(q, k=30):
    os_client = OpenSearch(OS_URL)
    body = {"size":k, "query":{"bool":{"should":[
        {"match":{"title":{"query":q,"boost":2.0}}},
        {"match":{"body":{"query":q}}},
        {"match":{"text":{"query":q}}}
    ]}}}
    r = os_client.search(index=OS_INDEX, body=body)
    hits=[]
    for h in r.get("hits",{}).get("hits",[]):
        src=h.get("_source",{}) or {}
        txt = src.get("body") or src.get("text") or ""
        hits.append({"id": h.get("_id",""), "text": txt, "source": src.get("source")})
    return hits

def classify(query, docs):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    device = "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"
    mdl.to(device)
    out=[]
    B=32
    for i in range(0, len(docs), B):
        chunk = docs[i:i+B]
        enc = tok([query]*len(chunk), [d["text"] for d in chunk], return_tensors="pt",
                  truncation=True, padding=True, max_length=128)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).tolist()
        for j,p in enumerate(preds):
            out.append({"stance": ID2LABEL[int(p)], "conf": float(probs[j][p]), **chunk[j]})
    return out

if __name__=="__main__":
    q = input("Query (e.g., 'cryptocurrency regulation'): ").strip() or "cryptocurrency regulation"
    k = 30
    docs = retrieve(q, k=k)
    if not docs:
        print("No hits in OpenSearch. Try a broader query.")
        raise SystemExit
    preds = classify(q, docs)
    c = Counter([p["stance"] for p in preds])
    print("\nCounts:", dict(c))
    for stance in ["favor","against","none"]:
        print(f"\n=== {stance.upper()} (top 3) ===")
        items = sorted([p for p in preds if p["stance"]==stance], key=lambda x: x["conf"], reverse=True)[:3]
        for it in items:
            preview = (it["text"][:140] + "…") if len(it["text"])>140 else it["text"]
            print(f"- {it['conf']:.2f} :: {preview}")