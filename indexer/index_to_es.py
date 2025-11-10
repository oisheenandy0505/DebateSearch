#!/usr/bin/env python3
import os, json, time
from pathlib import Path
from opensearchpy import OpenSearch, helpers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CORPUS_PATH = os.getenv("MERGED_PATH", str(PROCESSED / "corpus.jsonl"))

OS_URL = os.getenv("OS_URL", "http://localhost:9200")
INDEX = os.getenv("OS_INDEX", "debate_docs")

client = OpenSearch(OS_URL)

MAPPING = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {
            "analyzer": {"english_custom": {"type": "standard", "stopwords": "_english_"}}
        },
    },
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "body": {"type": "text", "analyzer": "english"},
            "text": {"type": "text", "analyzer": "english"},
            "source": {"type": "keyword"},
            "url": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "nsfw": {"type": "boolean"},
            "pii_masked": {"type": "boolean"},
            # optional older fields
            "subreddit": {"type": "keyword"},
            "created_utc": {"type": "date", "format": "epoch_second"},
            "score": {"type": "integer"},
            "target": {"type": "keyword"},
            "stance_gold": {"type": "keyword"},
        }
    },
}

def ensure_index():
    if not client.indices.exists(index=INDEX):
        client.indices.create(index=INDEX, body=MAPPING)
        print(f"🆕 Created index {INDEX}")
    else:
        print(f"ℹ️  Index {INDEX} already exists")

def actions(corpus_path: str):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- normalize fields before indexing ---
            doc.setdefault("title", doc.get("title") or doc.get("text","")[:80])
            doc.setdefault("text",  doc.get("text")  or doc.get("body",""))
            doc.setdefault("source", doc.get("source","web"))
            doc.setdefault("nsfw", bool(doc.get("nsfw", False)))
            doc.setdefault("pii_masked", bool(doc.get("pii_masked", False)))
            doc.setdefault("quality_score", float(doc.get("quality_score", 0.0)))
            doc.setdefault("stanceable", bool(doc.get("stanceable", True)))

            if doc["quality_score"] < 0.50 or not doc["stanceable"]:
                continue  # skip indexing the junk


            _id = doc.get("id") or f"doc-{i}"
            yield {
                "_index": INDEX,
                "_id": _id,
                "_source": doc,
            }

def main():
    ensure_index()
    start = time.time()
    total = 0
    for ok, info in helpers.streaming_bulk(
        client,
        actions(CORPUS_PATH),
        chunk_size=1000,
        max_retries=3,
        raise_on_error=False,
    ):
        total += 1
        if total % 10000 == 0:
            print(f"… indexed {total:,} docs")
    secs = time.time() - start
    print(f"Done. Indexed ~{total:,} docs in {secs:.1f}s → index={INDEX}")

if __name__ == "__main__":
    print(f"Using corpus: {CORPUS_PATH}")
    main()
