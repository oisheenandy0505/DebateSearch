#!/usr/bin/env python3
"""
Index data/processed/corpus.jsonl into OpenSearch index `debate_docs`,
including the quality_score field so we can boost high-quality comments.
"""

import json
from pathlib import Path
from typing import Iterable

from opensearchpy import OpenSearch, helpers

ROOT = Path(__file__).resolve().parents[1]   # DebateSearch/
DATA_DIR = ROOT / "data"                     # DebateSearch/data/
CORPUS_PATH = DATA_DIR / "processed" / "corpus.jsonl"

INDEX_NAME = "debate_docs"


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "admin"),
        scheme="http",
        verify_certs=False,
    )


def ensure_index(client: OpenSearch):
    if client.indices.exists(INDEX_NAME):
        print(f"Index {INDEX_NAME} already exists")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "body": {"type": "text"},
                "source": {"type": "keyword"},
                "subreddit": {"type": "keyword"},
                "created_utc": {"type": "date", "format": "epoch_second"},
                "score": {"type": "integer"},
                "quality_score": {"type": "float"},
                "target": {"type": "keyword"},
                "stance_gold": {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f" Created index {INDEX_NAME}")


def gen_actions(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = doc.get("id")
            if not doc_id:
                continue

            yield {
                "_index": INDEX_NAME,
                "_id": doc_id,
                "_source": doc,
            }


def main():
    if not CORPUS_PATH.exists():
        raise SystemExit(f"Missing corpus file: {CORPUS_PATH}")

    client = get_client()
    ensure_index(client)

    print(f"→ Indexing from {CORPUS_PATH}")
    success, failed = helpers.bulk(
        client,
        gen_actions(CORPUS_PATH),
        chunk_size=2000,
        request_timeout=120,
        stats_only=True,
    )

    print(f" Indexed {success:,} docs, failed={failed}")


if __name__ == "__main__":
    main()