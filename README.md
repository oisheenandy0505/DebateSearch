# Debate Search — Local Setup Guide

Here’s everything you need to get the app running locally — backend, frontend, and OpenSearch.

---

## Prerequisites

Make sure you have:
- **Docker + Docker Compose**
- **Python 3.12+**
- **Node.js 18+**
- **npm**
- **Git** (to clone and pull updates)

---

## Project Overview

This version is the **MVP (Milestone 1)** — a functional BM25-based search engine.  
Here’s what it currently includes:

- **Dockerized OpenSearch** — serves as our local search engine.
- **OpenSearch Dashboards** — for visualizing and exploring indexed data.
- **FastAPI backend** — handles indexing, searching, and connection to OpenSearch.
- **Vite/React frontend** — clean and fast UI for searching and displaying results.
- **Sample dataset** — preloaded JSON documents to test BM25 queries.

---

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/<your-repo>/DebateSearch.git
cd DebateSearch
```

### Start OpenSearch and Dashboards (Docker)
```bash
docker compose up -d
```

This will:

- Start **OpenSearch** → [http://localhost:9200](http://localhost:9200)  
- Start **OpenSearch Dashboards** → [http://localhost:5601](http://localhost:5601)

You can use the **OpenSearch Dashboards UI** to explore data and queries visually.

---

If Docker shows an *admin password* error:

```bash
docker compose down -v
docker compose up -d
```

> *(Our `docker-compose.yml` sets a default local admin password for OpenSearch.)*

---

To verify both containers are running:

```bash
docker ps
```

**Expected output:**
| CONTAINER ID | NAME                  | STATUS        | PORTS                   |
|--------------|-----------------------|---------------|-------------------------|
| xxxxxx       | opensearch            | Up (healthy)  | 0.0.0.0:9200->9200/tcp  |
| xxxxxx       | opensearch-dashboards | Up            | 0.0.0.0:5601->5601/tcp  |

---

## Backend Setup (FastAPI)

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### Create a `.env` file inside `backend/`:

```ini
OS_URL=http://localhost:9200
OS_INDEX=debate_docs
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```
### Run the backend server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Visit the FastAPI interactive docs at: http://localhost:8000/docs

---

## Seed Sample Data

To verify everything works, index the sample dataset:

```bash
python - <<'PY'
import json, requests
docs = [json.loads(l) for l in open('data/sample_docs.jsonl')]
print(requests.post('http://127.0.0.1:8000/index', json=docs).json())
PY
```

You should see something like:

```json
{"indexed": 5}
```

You can now view these documents in **OpenSearch Dashboards → Discover tab → select `debate_docs` index.**

---

## Frontend Setup (Vite/React)

```bash
cd ../frontend
npm install
npm run dev
```

The app should open automatically at:  
http://localhost:5173

If not, open it manually in your browser.

**If you get a CORS error:**

- Make sure the backend is running on port **8000**.
- Confirm your `.env` has `CORS_ORIGINS` including both origins:

```ini
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```
## Expected Behavior

Once both backend and frontend are running:

1. Visit **http://localhost:5173**
2. Type a query like **cryptocurrency regulation**
3. Click **Search**

You should see results ranked by **BM25 score**:

### Debate Search (BM25)

**A balanced framework for cryptocurrency oversight**  
→ Proposes licensing exchanges while preserving DeFi experimentation under sandboxes.  
`[paper | Score: 2.93]`

**Crypto regulation will curb scams, experts argue**  
→ Analysts claim tighter rules reduce fraud and stabilize markets.  
`[news | Score: 2.47]`

## Troubleshooting

| Problem | Fix |
|----------|------|
| **Search failed** | Backend isn’t running — start it with `uvicorn app:app --reload`. |
| **CORS error** | Add `CORSMiddleware` in FastAPI or ensure `.env` has correct `CORS_ORIGINS`. |
| **Docker password issue** | Add `OPENSEARCH_INITIAL_ADMIN_PASSWORD` in `docker-compose.yml` or run `docker compose down -v`. |
| **pydantic-core build failed** | Use **Python 3.12** (Python 3.14+ isn’t supported yet). |
| **Nothing in OpenSearch** | Re-run the indexing step and check **Dashboards → “debate_docs”** index. |
