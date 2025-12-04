# DebateSearch — Setup & Run (End-to-End)

---

## Prerequisites

Make sure you have:
- **Docker + Docker Compose**
- **Python 3.12+**
- **Node.js 18+**
- **npm**
- **Git** (to clone and pull updates)

---

## Clone Repository

```bash
git clone https://github.com/oisheenandy0505/DebateSearch.git
cd DebateSearch
```
---

## Start OpenSearch (Docker)

From the repo root (DebateSearch/):
```bash
docker compose up -d
```

Check OpenSearch is up:
```bash
curl -s "http://localhost:9200"
```
If you see JSON with cluster info, it’s good.

---

## Set up the backend (FastAPI)

From repo root:
```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Start the backend API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Now the API should be live at: http://localhost:8000

Leave this process running. 

---

## Set up the frontend
Open a new terminal window/tab.

From repo root (DebateSearch/):
```bash
cd frontend
npm install
npm run dev
```

By default Vite will serve at: http://localhost:5173

Open that in the browser and you should see the DebateSearch UI.

Type something like cryptocurrency regulation and hit Search — it should hit the FastAPI backend and show results.

