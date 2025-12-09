# DebateSearch

DebateSearch is an end-to-end system that surfaces pro/con evidence for any topic by combining:

- **OpenSearch retrieval** over a curated corpus (SemEval Twitter + high-signal Reddit threads).
- **A DistilBERT stance classifier** that labels each document as favor / against / neutral, with optional LoRA adapters and reweighted training regimes.
- **A FastAPI backend** that pipelines retrieval → stance classification → clustering → PII redaction.
- **A React frontend (Vite)** that visualizes the debate with grid/compare modes, source filters, NSFW toggles, and saved cards.

## Theory & Approach

1. **Corpus construction**
   - SemEval 2016 Task 6 tweets provide gold-labeled stance exemplars; we treat them as high-quality support/opposition ground truth.
   - Reddit comments are harvested (Pushshift or HF streaming), aggressively filtered for length, content density, NSFW, and deduplicated using SimHash. Each comment receives a quality score so ranking can favor substantive posts.
2. **Quality gating & deduping**
   - `backend/utils/quality.py` computes length/density/cue-based stance signal, coarse NSFW flags, and a 64-bit SimHash bucketed dedupe.
   - This gate is used by **all** ingest scripts (`scripts/prepare_reddit.py`, `scripts/prepare_semeval.py`) and the live backend before returning snippets.
3. **Modeling**
   - Base trainer (`backend/ml/train_stance.py`) fine-tunes DistilBERT on SemEval topics.
   - `train_stance_weighted.py` adds inverse-frequency class weighting for better macro-F1.
   - `train_stance_lora.py` trains LoRA adapters with class weighting + optional subsampling for quick iterations on laptops, then merges adapters back into a standalone model.
4. **Serving**
   - FastAPI (`backend/app.py`) loads the fine-tuned model, hits OpenSearch for retrieval, classifies each result, clusters by stance, and redacts obvious PII (emails, phones, IPs, @handles).
   - Simple NSFW heuristics remove explicit content unless `nsfw_ok` is passed.
5. **Frontend**
   - React/Vite app (`frontend/src/App.jsx`) POSTs `/search_clustered`, visualizes clusters, offers saved items, density toggles, and source filters.
6. **Retrieval**
   - OpenSearch sticks with BM25 for every text field. The `multi_match` query hits `body`, `title`, and `target`, while Lucene’s default BM25 handles TF saturation and built-in length normalization (the smoothing component). `_search_raw` layers on a `function_score` boost tied to `quality_score` (factor 1.2, missing 0.5), and results below `min_quality` are filtered out before stance classification so low-quality Reddit fragments never reach the UI.

## Repository Layout

```
backend/             FastAPI service + ML utilities
frontend/            React UI (Vite)
indexer/             Corpus construction + OpenSearch indexing
scripts/             Data prep, Reddit streaming, smoke tests
models/stance_distilbert   Fine-tuned model artifacts
data/processed       Cleaned datasets, corpus.jsonl
```

## Getting Started

### 1. Prerequisites

- Docker + Docker Compose (for OpenSearch)
- Python 3.12+
- Node.js 18+ with npm
- Git

### 2. Clone

```bash
git clone https://github.com/oisheenandy0505/DebateSearch.git
cd DebateSearch
```

## Data Pipeline

1. **Fetch Reddit data**
   - `python -m scripts.fetch_hf_reddit_stream` streams pushshift comments, filters with `is_junk_body`, computes heuristic `quality_score`, and writes `data/processed/reddit_clean.jsonl`.
2. **Prepare Reddit exports**
   - `python -m scripts.prepare_reddit --glob "data/raw/kaggle/reddit*.jsonl"` masks PII, runs the shared `quality_gate`, dedupes via SimHash buckets, and emits normalized JSONL.
3. **Prepare SemEval**
   - `python -m scripts.prepare_semeval` ingests TSV/TXT files from `data/raw/semeval/`, masks PII, applies the lenient quality gate, and produces stratified train/dev/test JSONL splits.
4. **Build final corpus**
   - `python -m indexer.build_corpus` merges `semeval_clean.jsonl` + `reddit_clean.jsonl` into `data/processed/corpus.jsonl`, preserving `quality_score`, `target`, and `stance_gold`.
5. **Index to OpenSearch**
   - Ensure Docker OpenSearch is running (next section), then `python -m indexer.index_to_es`.

### Training

The serving checkpoint comes from the LoRA trainer:

```bash
cd backend
python -m backend.ml.train_stance_lora
```

This mirrors the weighted trainer’s macro-F1 orientation while only updating adapters, which keeps memory/compute requirements modest. Both `train_stance.py` and `train_stance_weighted.py` remain for reference or ablation studies, but LoRA is the default workflow.

## Running the Stack

### 1. OpenSearch via Docker Compose

```bash
docker compose up -d
curl -s http://localhost:9200
```

### 2. Backend (FastAPI)

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend (React/Vite)

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Now search for a topic like “abortion” — the frontend will POST to `http://localhost:8000/search_clustered`, and you should see favor/against clusters with PII redacted.

## Smoke Tests & Diagnostics

- `python -m scripts.smoke_tests`: checks `quality_gate` and hits `/health` (make sure backend server is running).
- `curl http://localhost:8000/health`: confirms OpenSearch connectivity and doc counts.
- `python -m scripts.prepare_reddit --limit 500 --verbose`: quick sanity run of the Reddit cleaner.

## Notes

- Set `BACKEND_DEBUG_LOG_PATHS=1` when launching `uvicorn` to print resolved project pathss.
- PII redaction (`backend/ml/pii_utils.py`) masks emails, phones, IPv4 addresses, SSNs, and @handles before returning data to the UI.
- All ingest scripts rely on the shared `quality_gate`, ensuring consistent heuristics across offline prep and online serving.
