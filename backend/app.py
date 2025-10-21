import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from opensearchpy import OpenSearch
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OS_URL = os.getenv("OS_URL", "http://localhost:9200")
OS_INDEX = os.getenv("OS_INDEX", "debate_docs")

client = OpenSearch(OS_URL)

app = FastAPI(title="Debate Search API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAPPING = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {
            "analyzer": {
                "english_custom": {"type": "standard", "stopwords": "_english_"}
            }
        },
    },
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "body": {"type": "text", "analyzer": "english"},
            "source": {"type": "keyword"},
            "url": {"type": "keyword"},
            "timestamp": {"type": "date"}
        }
    },
}

def ensure_index():
    if not client.indices.exists(index=OS_INDEX):
        client.indices.create(index=OS_INDEX, body=MAPPING)

class DocIn(BaseModel):
    id: str
    url: str
    title: str
    body: str
    source: str
    timestamp: str

class SearchRespHit(BaseModel):
    id: str
    title: str
    body: str
    url: str
    source: str
    score: float

@app.on_event("startup")
def _startup():
    ensure_index()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/index")
def index_docs(docs: List[DocIn]):
    body = []
    for d in docs:
        body.extend([
            {"index": {"_index": OS_INDEX, "_id": d.id}},
            d.model_dump(),
        ])
    client.bulk(body=body, refresh=True)
    return {"indexed": len(docs)}

@app.get("/search", response_model=List[SearchRespHit])
def search(q: str = Query(..., min_length=1), k: int = 10):
    query = {
        "bool": {
            "should": [
                {"match": {"title": {"query": q, "boost": 2.0}}},
                {"match": {"body": {"query": q}}}
            ]
        }
    }
    res = client.search(index=OS_INDEX, body={"size": k, "query": query})
    hits = []
    for h in res["hits"]["hits"]:
        src = h["_source"]
        hits.append(
            SearchRespHit(
                id=h["_id"], title=src["title"], body=src["body"],
                url=src["url"], source=src["source"], score=h["_score"]
            )
        )
    return hits