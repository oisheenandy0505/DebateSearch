#!/usr/bin/env python3
"""
FastAPI backend for DebateSearch.

Endpoints
---------
GET  /health
GET  /search_clustered
POST /search_clustered

It talks to:
  - OpenSearch index "debate_docs" for retrieval
  - DistilBERT stance model in ../models/stance_distilbert for clustering
and:
  - redacts obvious PII before returning snippets.
"""

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.ml.pii_utils import redact_text

# -------------------------------------------------------------------
# Env & constants
# -------------------------------------------------------------------

load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
MODEL_DIR = PROJECT_ROOT / "models" / "stance_distilbert"

print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DEBUG] MODEL_DIR = {MODEL_DIR}")

OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_USER = os.getenv("OS_USER", "admin")
OS_PASS = os.getenv("OS_PASS", "admin")
INDEX_NAME = os.getenv("OS_INDEX", "debate_docs")

FILTER_NSFW_DEFAULT = os.getenv("FILTER_NSFW_DEFAULT", "true").lower() == "true"

MAX_K = 100
BATCH_SIZE = 16
MAX_SEQ_LEN = 256

# crude NSFW heuristic to downrank or filter
NSFW_RE = re.compile(r"\b(nsfw|porn|sex|xxx|onlyfans)\b", re.IGNORECASE)

# -------------------------------------------------------------------
# Global state: OpenSearch client + stance model
# -------------------------------------------------------------------

os_client: Optional[OpenSearch] = None
stance_tokenizer: Optional[AutoTokenizer] = None
stance_model: Optional[AutoModelForSequenceClassification] = None
ID2LABEL: Dict[int, str] = {0: "support", 1: "oppose", 2: "neutral"}


def get_os_client() -> OpenSearch:
    global os_client
    if os_client is None:
        os_client = OpenSearch(
            hosts=[{"host": OS_HOST, "port": OS_PORT}],
            http_auth=(OS_USER, OS_PASS),
            scheme="http",
            verify_certs=False,
        )
    return os_client


def load_stance_model():
    global stance_tokenizer, stance_model, ID2LABEL

    if stance_model is not None:
        return

    if not MODEL_DIR.exists():
        print(f"[WARN] stance model dir not found: {MODEL_DIR}")
        return

    print(f"[OK] loading stance model from {MODEL_DIR}")
    stance_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    stance_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    stance_model.eval()

    # device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    stance_model.to(device)
    stance_model.device_name = device  # type: ignore[attr-defined]

    # read label mapping if present
    meta_path = MODEL_DIR / "label_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open() as f:
                meta = json.load(f)
            raw = meta.get("id2label") or {}
            # keys may be strings
            ID2LABEL = {int(k): v for k, v in raw.items()}
        except Exception as e:
            print(f"[WARN] could not load label_meta.json: {e}")
    print(f"[OK] stance labels: {ID2LABEL}")


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class SearchReq(BaseModel):
    query: str
    k: int = 30
    nsfw_ok: Optional[bool] = None
    only_stanceable: bool = False
    min_quality: float = 0.3


class ClusterItem(BaseModel):
    id: str
    title: str
    body: str
    source: str
    subreddit: Optional[str] = None
    created_utc: Optional[int] = None
    score: Optional[float] = None
    quality_score: Optional[float] = None
    stance_label: Optional[str] = None
    stance_confidence: Optional[float] = None


class StanceCluster(BaseModel):
    stance: str  # "favor" | "against" | "none"
    items: List[ClusterItem]


class ClusteredResponse(BaseModel):
    query: str
    clusters: List[StanceCluster]
    meta: Dict[str, Any]


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="DebateSearch backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # local dev; can be narrowed via env later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # warm up connections
    client = get_os_client()
    try:
        client.indices.exists(INDEX_NAME)
        print(f"[OK] index {INDEX_NAME} reachable")
    except Exception as e:
        print(f"[WARN] could not reach OpenSearch: {e}")

    load_stance_model()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _is_nsfw(text: str) -> bool:
    return bool(NSFW_RE.search(text or ""))


def _search_raw(query: str, k: int, min_quality: float, nsfw_ok: bool) -> List[Dict[str, Any]]:
    client = get_os_client()
    size = min(max(k, 1), MAX_K)

    must_query: Dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": ["body^2", "title", "target"],
        }
    }

    filters: List[Dict[str, Any]] = []
    if min_quality is not None:
        filters.append({"range": {"quality_score": {"gte": min_quality}}})

    body: Dict[str, Any] = {
        "size": size,
        "_source": [
            "id",
            "title",
            "body",
            "source",
            "subreddit",
            "created_utc",
            "score",
            "quality_score",
            "target",
            "stance_gold",
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [must_query],
                        "filter": filters,
                    }
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "quality_score",
                            "factor": 1.2,
                            "missing": 0.5,
                        }
                    }
                ],
                "score_mode": "avg",
                "boost_mode": "sum",
            }
        },
    }

    try:
        res = client.search(index=INDEX_NAME, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenSearch error: {e}")

    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {})
        src["es_score"] = float(h.get("_score") or 0.0)
        if not nsfw_ok and _is_nsfw(src.get("body", "")):
            continue
        out.append(src)
    return out


@torch.inference_mode()
def _classify_stance(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run DistilBERT on (query, body) pairs and attach:
      - stance_label in {"favor","against","none"}
      - stance_confidence in [0,1]

    This version is slightly more *aggressive* at assigning favor/against:
      - lower confidence + margin thresholds,
      - raw "neutral" predictions still go to "none".
    """
    if not docs or stance_model is None or stance_tokenizer is None:
        return docs

    device = getattr(stance_model, "device_name", "cpu")  # type: ignore[attr-defined]

    texts = [d.get("body", "") for d in docs]
    results: List[Dict[str, Any]] = []

    # more permissive thresholds than before
    HIGH_CONF = 0.55          # was 0.7
    NEUTRAL_MARGIN = 0.10     # was 0.2

    for start in range(0, len(texts), BATCH_SIZE):
        batch_docs = docs[start:start + BATCH_SIZE]
        batch_bodies = texts[start:start + BATCH_SIZE]

        enc = stance_tokenizer(
            [query] * len(batch_bodies),
            batch_bodies,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = stance_model(**enc).logits  # [B, 3]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for doc, prob_vec in zip(batch_docs, probs):
            top_idx = int(prob_vec.argmax())
            top_prob = float(prob_vec[top_idx])

            # second-best prob
            sorted_probs = sorted(prob_vec, reverse=True)
            second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

            raw_label = ID2LABEL.get(top_idx, "neutral").lower()

            # default bucket
            stance = "none"

            # FORCE: if model explicitly calls it "neutral", treat as none
            if raw_label == "neutral":
                stance = "none"
            else:
                # model prefers support/oppose
                if top_prob >= HIGH_CONF and (top_prob - second_prob) >= NEUTRAL_MARGIN:
                    if raw_label == "support":
                        stance = "favor"
                    elif raw_label == "oppose":
                        stance = "against"
                    else:
                        stance = "none"
                else:
                    # borderline: still give it a side if it's clearly not neutral
                    # but margin is small. This pushes more into favor/against.
                    if top_prob >= 0.5 and raw_label in {"support", "oppose"}:
                        stance = "favor" if raw_label == "support" else "against"
                    else:
                        stance = "none"

            doc["stance_label"] = stance
            doc["stance_confidence"] = top_prob
            results.append(doc)

    return results

def _cluster_results(query: str, docs: List[Dict[str, Any]], only_stanceable: bool) -> ClusteredResponse:
    # classify
    docs = _classify_stance(query, docs)

    clusters: Dict[str, List[ClusterItem]] = {
        "favor": [],
        "against": [],
        "none": [],
    }

    for d in docs:
        stance = d.get("stance_label") or "none"
        if only_stanceable and stance == "none":
            continue

        # IMPORTANT: redact_text returns (clean_text, had_pii_flag) → unpack it
        clean_body, _had_pii = redact_text(d.get("body", "") or "")

        item = ClusterItem(
            id=str(d.get("id")),
            title=d.get("title") or "(untitled)",
            body=clean_body,
            source=d.get("source", "unknown"),
            subreddit=d.get("subreddit"),
            created_utc=d.get("created_utc"),
            score=d.get("score"),
            quality_score=d.get("quality_score"),
            stance_label=stance,
            stance_confidence=d.get("stance_confidence"),
        )
        clusters[stance].append(item)

    cluster_list: List[StanceCluster] = []
    for stance_key in ["favor", "against", "none"]:
        cluster_list.append(StanceCluster(stance=stance_key, items=clusters[stance_key]))

    meta = {
        "retrieved": len(docs),
        "classified": sum(len(c.items) for c in cluster_list),
    }
    return ClusteredResponse(query=query, clusters=cluster_list, meta=meta)


def _search_clustered_core(
    query: str,
    k: int,
    nsfw_ok: bool,
    min_quality: float,
    only_stanceable: bool,
) -> ClusteredResponse:
    query = (query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs = _search_raw(query, k, min_quality, nsfw_ok)
    if not docs:
        return ClusteredResponse(query=query, clusters=[], meta={"retrieved": 0, "classified": 0})

    return _cluster_results(query, docs, only_stanceable)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health")
def health():
    client = get_os_client()
    try:
        ping = client.ping()
        count = client.count(index=INDEX_NAME).get("count", 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenSearch error: {e}")
    return {
        "status": "ok" if ping else "degraded",
        "index": INDEX_NAME,
        "doc_count": count,
    }


@app.get("/search_clustered", response_model=ClusteredResponse)
def search_clustered_get(
    q: str = Query(..., min_length=1),
    k: int = 30,
    nsfw_ok: bool = FILTER_NSFW_DEFAULT,
    min_quality: float = 0.4,
    only_stanceable: bool = False,
):
    return _search_clustered_core(q, k, nsfw_ok, min_quality, only_stanceable)


@app.post("/search_clustered", response_model=ClusteredResponse)
def search_clustered_post(req: SearchReq = Body(...)):
    nsfw_ok = FILTER_NSFW_DEFAULT if req.nsfw_ok is None else req.nsfw_ok
    return _search_clustered_core(
        req.query,
        req.k,
        nsfw_ok,
        req.min_quality,
        req.only_stanceable,
    )