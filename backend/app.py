# backend/app.py
import os, json, re
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.ml.pii_utils import redact_text

# ---------- Env & constants ----------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OS_URL   = os.getenv("OS_URL", "http://localhost:9200")
OS_INDEX = os.getenv("OS_INDEX", "debate_docs")

STANCE_MODEL_DIR = os.getenv("STANCE_MODEL_DIR", "models/stance_distilbert")
CUES_PATH        = os.getenv("STANCE_CUES_PATH", "models/stance_cues.json")

# Bias 'none' slightly down; tune [-0.25..-0.6]
NONE_LOGIT_BIAS  = float(os.getenv("STANCE_NONE_LOGIT_BIAS", "-0.35"))
# Match training
MAX_LEN_INFER    = int(os.getenv("STANCE_MAX_LEN_INFER", "192"))
# Default filter behavior
FILTER_NSFW_DEFAULT = os.getenv("FILTER_NSFW_DEFAULT", "true").lower() in {"1","true","yes"}
MASK_PII_AT_SERVE   = os.getenv("MASK_PII_AT_SERVE", "true").lower() in {"1","true","yes"}

# ---------- OpenSearch ----------
client = OpenSearch(OS_URL)

# ---------- FastAPI ----------
app = FastAPI(title="Debate Search API", version="0.4.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- Mapping (adds nsfw, pii_masked, quality_score, stanceable) ----------
MAPPING = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {"analyzer": {"english_custom": {"type": "standard","stopwords":"_english_"}}}
    },
    "mappings": {
        "properties": {
            "title": {"type":"text","analyzer":"english"},
            "body":  {"type":"text","analyzer":"english"},
            "text":  {"type":"text","analyzer":"english"},
            "source":{"type":"keyword"},
            "url":   {"type":"keyword"},
            "timestamp":{"type":"date"},
            "nsfw": {"type":"boolean"},
            "pii_masked": {"type":"boolean"},
            "quality_score": {"type":"float"},
            "stanceable":    {"type":"boolean"},
            # legacy/extra fields you might have:
            "subreddit": {"type":"keyword"},
            "created_utc": {"type": "date", "format": "epoch_second"},
            "score": {"type": "integer"},
            "target": {"type": "keyword"},
            "stance_gold": {"type": "keyword"},
            # optional derived score
            "topicness": {"type":"float"}
        }
    },
}

def ensure_index():
    try:
        if not client.indices.exists(index=OS_INDEX):
            client.indices.create(index=OS_INDEX, body=MAPPING)
            print(f"[OK] created index {OS_INDEX}")
        else:
            print(f"[OK] index {OS_INDEX} exists")
    except Exception as e:
        print("[WARN] ensure_index:", e)

# ---------- Query helpers ----------
STOP = set("""
a an the and or of in on to for with by as at from is are was were be been being
this that these those it its into over under about against between during without
""".split())

TOPIC_SYNONYMS = {
    "abortion": [
        "abortion", "roe v wade", "dobbs", "planned parenthood",
        "pro-life", "prolife", "pro choice", "pro-choice",
        "heartbeat bill", "fetus", "reproductive rights"
    ],
    "cryptocurrency regulation": [
        "cryptocurrency regulation", "crypto regulation", "bitcoin regulation",
        "sec crypto", "coinbase sec", "binance", "stablecoin",
        "crypto ban", "crypto compliance", "ftx"
    ],
}

def _multi_match(query: str, fields=None, mm: str | None = None, operator: str = "and"):
    """Build a multi_match without emitting null keys (avoids 400 errors)."""
    if fields is None:
        fields = ["title^2", "body", "text"]
    mm_body: Dict[str, Any] = {
        "query": query,
        "fields": fields,
        "type": "most_fields",
        "operator": operator,
    }
    if mm is not None:
        mm_body["minimum_should_match"] = mm
    return {"multi_match": mm_body}

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower()) if t not in STOP]

def _best_topic(q: str) -> str | None:
    q_low = q.lower()
    for topic in TOPIC_SYNONYMS.keys():
        if topic in q_low:
            return topic
    return None

def build_query(q: str, strict: bool, min_quality: float | None, hide_nsfw: bool, only_stanceable: bool):
    """Return (query, rescore) ready for OpenSearch."""
    # filters (tolerant to missing fields)
    filt: List[Dict[str, Any]] = []
    if hide_nsfw:
        filt.append({
            "bool": {
                "should": [
                    {"term": {"nsfw": False}},
                    {"bool": {"must_not": {"exists": {"field": "nsfw"}}}}
                ],
                "minimum_should_match": 1
            }
        })
    if min_quality is not None:
        filt.append({
            "bool": {
                "should": [
                    {"range": {"quality_score": {"gte": float(min_quality)}}},
                    {"bool": {"must_not": {"exists": {"field": "quality_score"}}}}
                ],
                "minimum_should_match": 1
            }
        })
    if only_stanceable:
        filt.append({
            "bool": {
                "should": [
                    {"term": {"stanceable": True}},
                    {"bool": {"must_not": {"exists": {"field": "stanceable"}}}}
                ],
                "minimum_should_match": 1
            }
        })

    # core lexical match
    tokens = _tokenize(q)
    mm_val = None
    if not strict:
        # require ~70% of query terms if loose; otherwise AND all terms
        mm_val = f"{max(1, int(0.7 * len(tokens)))}" if tokens else "1"

    must_clause = _multi_match(
        q,
        fields=["title^3", "text^2", "body"],
        mm=mm_val,
        operator="and" if strict else "or",
    )

    # phrase boost (helps kill throwaways) + topic synonyms boost + topicness hint
    phrase_should = [{"match_phrase": {"text": {"query": q, "slop": 2}}}]
    topic_boost_should: List[Dict[str, Any]] = []
    topic = _best_topic(q)
    if topic and TOPIC_SYNONYMS.get(topic):
        for term in TOPIC_SYNONYMS[topic]:
            topic_boost_should.append({"match_phrase": {"text": {"query": term}}})

    query = {
        "bool": {
            "filter": filt,
            "must": must_clause,
            "should": phrase_should + topic_boost_should + [
                {"range": {"topicness": {"gte": 0.30}}}
            ],
        }
    }

    rescore = {
        "window_size": 200,
        "query": {
            "rescore_query": {"match_phrase": {"text": {"query": q, "slop": 1}}},
            "query_weight": 1.0,
            "rescore_query_weight": 2.0,
        },
    }
    return query, rescore

# ---------- Pydantic models ----------
class DocIn(BaseModel):
    id: str
    url: str | None = None
    title: str | None = None
    body: str | None = None
    text: str | None = None
    source: str | None = None
    timestamp: str | None = None
    nsfw: bool | None = None
    pii_masked: bool | None = None
    quality_score: float | None = None
    stanceable: bool | None = None
    meta: dict | None = None

class SearchReq(BaseModel):
    query: str
    k: int = 30
    nsfw_ok: bool | None = None      # override default NSFW filter
    min_quality: float = 0.5         # NEW
    only_stanceable: bool = True     # NEW
    strict: bool = True              # allow strict/loose from POST

class SearchRespHit(BaseModel):
    id: str
    title: str
    body: str
    url: str
    source: str
    score: float

class ClusterItem(BaseModel):
    id: str
    text: str
    source: str | None = None
    conf: float
    stance: str

class Cluster(BaseModel):
    stance: str
    items: List[ClusterItem]

class ClusteredResponse(BaseModel):
    query: str
    clusters: List[Cluster]
    meta: Dict[str, Any]

# ---------- Globals for stance model & cues ----------
tok = None
mdl = None
IDX2LBL = {0:"favor",1:"against",2:"none"}
LBL2IDX = {"favor":0,"against":1,"none":2}
STANCE_CUES = {"favor":{}, "against":{}}

def cue_score(text: str):
    low = text.lower()
    sf = sum(w for k,w in STANCE_CUES.get("favor",{}).items() if k in low)
    sa = sum(w for k,w in STANCE_CUES.get("against",{}).items() if k in low)
    return sf, sa

# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    ensure_index()
    global tok, mdl, STANCE_CUES
    try:
        tok = AutoTokenizer.from_pretrained(STANCE_MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_DIR)
        mdl.eval()
        print(f"[OK] stance model loaded from {STANCE_MODEL_DIR}")
    except Exception as e:
        print(f"[WARN] stance model not loaded: {e}")

    try:
        p = Path(CUES_PATH)
        if p.exists():
            STANCE_CUES = json.loads(p.read_text())
            print(f"[OK] loaded stance cues: "
                  f"{len(STANCE_CUES.get('favor',{}))} favor / "
                  f"{len(STANCE_CUES.get('against',{}))} against")
        else:
            print("[WARN] stance_cues.json not found; tie-breaker still runs.")
    except Exception as e:
        print("[WARN] failed to load stance cues:", e)

# ---------- Health ----------
@app.get("/health")
@app.get("/healthz")
def healthz():
    try:
        return {"ok": bool(client.ping())}
    except Exception:
        return {"ok": False}

# ---------- Ingest ----------
@app.post("/index")
def index_docs(docs: List[DocIn]):
    body=[]
    for d in docs:
        src = d.model_dump()

        # Normalize/ensure text fields
        if src.get("text") and not src.get("body"):
            src["body"] = src["text"]
        if not src.get("title") and (src.get("body") or src.get("text")):
            src["title"] = (src.get("body") or src.get("text"))[:80]

        # Ensure booleans exist
        src["nsfw"] = bool(src.get("nsfw", False))
        src["pii_masked"] = bool(src.get("pii_masked", False))

        # Provide safe defaults for new filter fields
        if src.get("quality_score") is None:
            src["quality_score"] = 0.0
        if src.get("stanceable") is None:
            src["stanceable"] = True

        _id = src.get("id")
        body.extend([{"index":{"_index":OS_INDEX,"_id":_id}}, src])

    client.bulk(body=body, refresh=True)
    return {"indexed": len(docs)}

# ---------- Retrieval ----------
def _do_search(
    q: str,
    k: int,
    nsfw_ok: bool,
    strict: bool = True,
    min_quality: float | None = None,
    only_stanceable: bool = True
) -> List[SearchRespHit]:
    if not q or k <= 0:
        raise HTTPException(400, "Invalid query or k")

    query, rescore = build_query(
        q=q,
        strict=strict,
        min_quality=min_quality,
        hide_nsfw=(not nsfw_ok),
        only_stanceable=only_stanceable,
    )

    try:
        body = {"size": k, "query": query}
        if rescore:
            body["rescore"] = rescore
        res = client.search(index=OS_INDEX, body=body)
    except Exception as e:
        raise HTTPException(502, f"OpenSearch error: {e}")

    hits: List[SearchRespHit] = []
    for h in res.get("hits",{}).get("hits",[]):
        src = h.get("_source",{}) or {}
        body_txt = src.get("body") or src.get("text") or ""
        if MASK_PII_AT_SERVE:
            body_txt, _ = redact_text(body_txt)
        title_txt = src.get("title") or (body_txt[:80] if body_txt else "")
        hits.append(SearchRespHit(
            id=h.get("_id",""),
            title=title_txt, body=body_txt,
            url=src.get("url","") or "",
            source=src.get("source","") or (src.get("meta",{}) or {}).get("subreddit","") or "",
            score=float(h.get("_score", 0.0)),
        ))
    return hits

@app.get("/search", response_model=List[SearchRespHit])
def search_get(
    q: str = Query(..., min_length=1),
    k: int = 10,
    nsfw_ok: bool = FILTER_NSFW_DEFAULT,
    strict: bool = True,
    min_quality: float | None = 0.3,
    only_stanceable: bool = True
):
    return _do_search(q, k, nsfw_ok, strict, min_quality, only_stanceable)

@app.post("/search", response_model=List[SearchRespHit])
def search_post(req: SearchReq):
    nsfw_ok = FILTER_NSFW_DEFAULT if req.nsfw_ok is None else req.nsfw_ok
    return _do_search(
        q=req.query,
        k=req.k,
        nsfw_ok=nsfw_ok,
        strict=req.strict,
        min_quality=req.min_quality,
        only_stanceable=req.only_stanceable,
    )

# ---------- Stance classification ----------
def _classify_stance(query: str, docs: List[str]) -> List[Dict[str, Any]]:
    if not (tok and mdl) or not docs:
        return [{"stance":"none","conf":0.0} for _ in docs]
    device = ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    out=[]; B=32
    with torch.no_grad():
        for i in range(0,len(docs),B):
            chunk = docs[i:i+B]
            enc = tok([query]*len(chunk), chunk, return_tensors="pt",
                      truncation=True, padding=True, max_length=MAX_LEN_INFER)
            enc = {k:v.to(device) for k,v in enc.items()}
            logits = mdl(**enc).logits
            if NONE_LOGIT_BIAS != 0.0:
                logits = logits + torch.tensor([0.0,0.0,NONE_LOGIT_BIAS], device=logits.device)
            logits = logits / 0.95
            probs  = torch.softmax(logits, dim=1)
            for j, text in enumerate(chunk):
                p = probs[j]
                top2v, top2i = torch.topk(p, 2)
                pred = int(top2i[0].item())
                # light, cue-based nudge when 'none' is uncertain
                if pred == LBL2IDX["none"] and (top2v[0]-top2v[1]) <= 0.12:
                    sf, sa = cue_score(text)
                    if (sa - sf) >= 0.15: pred = LBL2IDX["against"]
                    elif (sf - sa) >= 0.15: pred = LBL2IDX["favor"]
                out.append({"stance": IDX2LBL[pred], "conf": float(p[pred].item())})
    return out

# ---------- Clustered search ----------
def _search_clustered_core(
    q: str,
    k: int,
    nsfw_ok: bool,
    min_quality: float = 0.0,
    only_stanceable: bool = False,
):
    if not q:
        raise HTTPException(400, "Empty query")

    # filters (tolerant)
    filt: List[Dict[str, Any]] = []
    if min_quality > 0:
        filt.append({"range": {"quality_score": {"gte": float(min_quality)}}})
    if not nsfw_ok:
        filt.append({"term": {"nsfw": False}})
    if only_stanceable:
        filt.append({"term": {"stanceable": True}})

    must_clause = _multi_match(q, ["title^2", "body", "text"], mm=None, operator="and")
    phrase_should = [
        {"match_phrase": {"title": {"query": q, "boost": 2.0}}},
        {"match_phrase": {"body":  {"query": q, "boost": 1.5}}},
        {"match_phrase": {"text":  {"query": q, "boost": 1.5}}},
    ]
    topic_boost_should = [{"range": {"topicness": {"gte": 0.30}}}]

    query = {"bool": {"filter": filt, "must": must_clause, "should": phrase_should + topic_boost_should}}

    try:
        res = client.search(index=OS_INDEX, body={"size": max(1, min(200, k)), "query": query})
    except Exception as e:
        raise HTTPException(502, f"OpenSearch error: {e}")

    hits = (res.get("hits") or {}).get("hits") or []
    if not hits:
        return {"query": q, "clusters": [], "meta": {"retrieved": 0, "classified": 0}}

    # collect + PII-mask
    docs, ids, sources = [], [], []
    for h in hits:
        src = h.get("_source") or {}
        txt = src.get("body") or src.get("text") or ""
        if MASK_PII_AT_SERVE:
            txt, _ = redact_text(txt)
        docs.append(txt)
        ids.append(h.get("_id", ""))
        sources.append(src.get("source") or (src.get("meta") or {}).get("subreddit") or "")

    # drop ultra-short + near-duplicates
    seen = set()
    docs2, ids2, sources2 = [], [], []
    for i, txt in enumerate(docs):
        t = (txt or "").strip()
        if len(t) < 40:
            continue
        sig = (t[:120], len(t))
        if sig in seen:
            continue
        seen.add(sig)
        docs2.append(t); ids2.append(ids[i]); sources2.append(sources[i])

    docs, ids, sources = docs2, ids2, sources2
    if not docs:
        return {"query": q, "clusters": [], "meta": {"retrieved": 0, "classified": 0}}

    preds = _classify_stance(q, docs)

    buckets = {"favor": [], "against": [], "none": []}
    for i, txt in enumerate(docs):
        pr = preds[i]
        buckets[pr["stance"]].append({
            "id": ids[i],
            "text": txt,
            "source": sources[i],
            "conf": pr["conf"],
            "stance": pr["stance"],
        })

    clusters = []
    for stance in ("favor", "against", "none"):
        items = sorted(buckets[stance], key=lambda x: x["conf"], reverse=True)
        clusters.append({"stance": stance, "items": items})

    return {"query": q, "clusters": clusters, "meta": {"retrieved": len(docs), "classified": len(docs)}}

@app.get("/search_clustered", response_model=ClusteredResponse)
def search_clustered_get(
    q: str = Query(..., min_length=1),
    k: int = 30,
    nsfw_ok: bool = FILTER_NSFW_DEFAULT,
    min_quality: float | None = 0.3,
    only_stanceable: bool = True
):
    return _search_clustered_core(q, k, nsfw_ok, float(min_quality or 0.0), only_stanceable)

@app.post("/search_clustered", response_model=ClusteredResponse)
def search_clustered_post(req: SearchReq = Body(...)):
    nsfw_ok = FILTER_NSFW_DEFAULT if req.nsfw_ok is None else req.nsfw_ok
    return _search_clustered_core(req.query, max(req.k,30), nsfw_ok, float(req.min_quality or 0.0), req.only_stanceable)