import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from opensearchpy import OpenSearch
from fastapi.middleware.cors import CORSMiddleware

# --- Stance model imports ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- Env & OS client ----------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OS_URL = os.getenv("OS_URL", "http://localhost:9200")
OS_INDEX = os.getenv("OS_INDEX", "debate_docs")

STANCE_MODEL_DIR = os.getenv("STANCE_MODEL_DIR", "models/stance_distilbert")
# Optional tiny inference bias to reduce "all none" dominance; e.g., -0.25
BIAS_NONE_LOGIT = float(os.getenv("STANCE_NONE_LOGIT_BIAS", "-0.0"))

client = OpenSearch(OS_URL)

app = FastAPI(title="Debate Search API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic models (existing) ----------------
class DocIn(BaseModel):
    id: str
    url: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[str] = None
    text: Optional[str] = None  # many pipelines produce "text"

class SearchRespHit(BaseModel):
    id: str
    title: str
    body: str
    url: str
    source: str
    score: float

class SearchReq(BaseModel):
    query: str
    k: int = 10

# ---------------- Clustered response models ----------------
class ClusterItem(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    conf: float
    stance: str

class Cluster(BaseModel):
    stance: str
    items: List[ClusterItem]

class ClusteredResponse(BaseModel):
    query: str
    clusters: List[Cluster]
    meta: Dict[str, Any]

# ---------------- Stance model state ----------------
tok = None
mdl = None
ID2LABEL = {0: "favor", 1: "against", 2: "none"}

def _torch_device() -> str:
    return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

# ---------------- Startup ----------------
@app.on_event("startup")
def _startup():
    # Ensure OS reachable (non-fatal if not)
    try:
        client.ping()
    except Exception:
        pass

    # Load stance model (non-fatal; /search works even without it)
    global tok, mdl
    try:
        tok = AutoTokenizer.from_pretrained(STANCE_MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_DIR)
        mdl.eval()
        # Put model on device once
        mdl.to(_torch_device())
        print(f"[OK] stance model loaded from {STANCE_MODEL_DIR} on {_torch_device()}")
    except Exception as e:
        print(f"[WARN] stance model not loaded: {e}")

# ---------------- Health ----------------
@app.get("/health")
@app.get("/healthz")
def healthz():
    try:
        return {"ok": bool(client.ping())}
    except Exception:
        return {"ok": False}

# ---------------- Indexing ----------------
@app.post("/index")
def index_docs(docs: List[DocIn]):
    body = []
    for d in docs:
        src = d.model_dump()
        if src.get("text") and not src.get("body"):
            src["body"] = src["text"]
        if not src.get("title") and src.get("body"):
            src["title"] = src["body"][:80]
        _id = src.get("id")
        body.extend([{"index": {"_index": OS_INDEX, "_id": _id}}, src])
    client.bulk(body=body, refresh=True)
    return {"indexed": len(docs)}

# ---------------- Flat search (existing) ----------------
def _do_search(q: str, k: int) -> List[SearchRespHit]:
    if not q or k <= 0:
        raise HTTPException(status_code=400, detail="Invalid query or k")
    query = {
        "bool": {
            "should": [
                {"match": {"title": {"query": q, "boost": 2.0}}},
                {"match": {"body":  {"query": q}}},
                {"match": {"text":  {"query": q}}},
            ]
        }
    }
    try:
        res = client.search(index=OS_INDEX, body={"size": k, "query": query})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")

    hits: List[SearchRespHit] = []
    for h in res.get("hits", {}).get("hits", []):
        src = h.get("_source", {}) or {}
        body_txt = src.get("body") or src.get("text") or ""
        title_txt = src.get("title") or (body_txt[:80] if body_txt else "")
        hits.append(
            SearchRespHit(
                id=h.get("_id", ""),
                title=title_txt,
                body=body_txt,
                url=src.get("url", "") or "",
                source=src.get("source", "") or (src.get("meta", {}) or {}).get("subreddit", "") or "",
                score=float(h.get("_score", 0.0)),
            )
        )
    return hits

@app.get("/search", response_model=List[SearchRespHit])
def search_get(q: str = Query(..., min_length=1), k: int = 10):
    return _do_search(q, k)

@app.post("/search", response_model=List[SearchRespHit])
def search_post(req: SearchReq):
    return _do_search(req.query, req.k)

# ---------------- Stance classification helpers ----------------
def _classify_stance(query: str, docs: List[str]) -> List[Dict[str, Any]]:
    """Return list of dicts with 'stance' and 'conf' for each doc."""
    if not docs:
        return []
    if tok is None or mdl is None:
        # Model not loaded: default to 'none'
        return [{"stance": "none", "conf": 0.0} for _ in docs]

    enc = tok([query] * len(docs), docs, return_tensors="pt",
              truncation=True, padding=True, max_length=128)
    device = _torch_device()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = mdl(**enc).logits
        if BIAS_NONE_LOGIT != 0.0:
            # logits are ordered [favor, against, none] for this head
            bias = torch.tensor([0.0, 0.0, -0.25], device=logits.device)  # [favor, against, none]
            logits = logits + bias
            probs = torch.softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).tolist()

    out: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        out.append({"stance": ID2LABEL[int(p)], "conf": float(probs[i][p])})
    return out

def _search_clustered_core(q: str, k: int = 30) -> ClusteredResponse:
    query = {
        "bool": {
            "should": [
                {"match": {"title": {"query": q, "boost": 2.0}}},
                {"match": {"body":  {"query": q}}},
                {"match": {"text":  {"query": q}}},
            ]
        }
    }
    try:
        res = client.search(index=OS_INDEX, body={"size": max(1, min(200, k)), "query": query})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        return ClusteredResponse(query=q, clusters=[], meta={"retrieved": 0, "classified": 0})

    ids, texts, sources = [], [], []
    for h in hits:
        src = h.get("_source", {}) or {}
        txt = src.get("body") or src.get("text") or ""
        ids.append(h.get("_id", ""))
        texts.append(txt)
        sources.append(src.get("source") or (src.get("meta", {}) or {}).get("subreddit"))

    preds = _classify_stance(q, texts)

    buckets: Dict[str, List[ClusterItem]] = {"favor": [], "against": [], "none": []}
    for i, txt in enumerate(texts):
        item = ClusterItem(
            id=ids[i],
            text=txt,
            source=sources[i],
            conf=preds[i]["conf"],
            stance=preds[i]["stance"],
        )
        buckets[item.stance].append(item)

    clusters: List[Cluster] = []
    for stance in ["favor", "against", "none"]:
        items_sorted = sorted(buckets[stance], key=lambda x: x.conf, reverse=True)
        clusters.append(Cluster(stance=stance, items=items_sorted))

    return ClusteredResponse(
        query=q,
        clusters=clusters,
        meta={"retrieved": len(texts), "classified": len(texts)}
    )

# ---------------- Clustered routes ----------------
class ClusterReq(BaseModel):
    query: str
    k: int = 30

@app.get("/search_clustered", response_model=ClusteredResponse)
def search_clustered_get(q: str = Query(..., min_length=1), k: int = 30):
    return _search_clustered_core(q, k)

@app.post("/search_clustered", response_model=ClusteredResponse)
def search_clustered_post(req: ClusterReq):
    return _search_clustered_core(req.query, req.k)