# parkrun Insights — Backend

Python FastAPI backend for the parkrun Insights Platform. Ingests survey CSV/XLSX exports, runs a compounding LLMWiki knowledge pipeline, and answers natural-language questions via a parallel multi-source query agent.

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- A Supabase project with the credentials in `.env`
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 2. Install dependencies

```bash
cd backend
pip install -r requirements-dev.txt
```

> **First run with local embeddings:** sentence-transformers will download the
> `nomic-ai/nomic-embed-text-v1.5` model (~270 MB) on the first embedding call.
> Set `EMBEDDING_PROVIDER=together` in `.env` to use the API instead.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 4. Apply database migrations

You need the database password from Supabase Dashboard → Project Settings → Database:

```bash
# Add SUPABASE_DB_URL to .env first, then:
pip install psycopg2-binary
python scripts/apply_migrations.py
```

Alternatively, copy each file from `migrations/` into the Supabase Dashboard SQL Editor and run them in order (001 → 008).

### 5. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/api/docs

### 6. Start the background worker (separate terminal)

```bash
python -m app.worker.pipeline
```

The worker polls the `ingestion_jobs` table every 10 seconds and processes pending jobs through the embedding → clustering → wiki pipeline.

---

## Running tests

```bash
pytest tests/ -v
```

Tests run entirely offline — no Supabase connection or API keys required. All external calls are mocked.

---

## Architecture

```
POST /api/ingest/upload
  │  Stage 1: Parse CSV/XLSX (handles SurveyMonkey 2-row headers)
  │  Stage 2: Write surveys + questions to DB
  │  Stage 3: Write responses + open-ended answers to DB
  │  → Creates ingestion_jobs row, returns immediately
  │
  └─► Background Worker
        Stage 4: Classify question types (Groq llama-3.1-8b-instant)
        Stage 5: Embed open-ended answers (nomic-embed-text-v1.5)
        Stage 6: HDBSCAN clustering + LLM cluster labelling
        Stage 7: LLMWiki ingest → wiki_pages updated

POST /api/chat
  │  Router: classify question type (Groq)
  │  Parallel: wiki_lookup + sql_query + [semantic_search + clusters]
  │  Synthesiser: merge → streaming SSE response (Groq llama-3.3-70b)
  └─► SSE stream to frontend
```

## Key files

| File | Purpose |
|---|---|
| `app/services/parser.py` | CSV/XLSX parsing; SurveyMonkey 2-row header support |
| `app/services/classifier.py` | LLM-based question type classification |
| `app/services/embedder.py` | nomic-embed-text embeddings (local or Together AI) |
| `app/services/clusterer.py` | HDBSCAN clustering + LLM cluster labelling |
| `app/services/wiki_maintainer.py` | LLMWiki ingest/lint via Groq tool calling |
| `app/services/query_agent.py` | Parallel retrieval + synthesis |
| `app/worker/pipeline.py` | Background job queue processor |
| `wiki/SCHEMA.md` | LLMWiki schema document (version-controlled) |
| `migrations/` | PostgreSQL migration SQL files (run in order 001→008) |

## LLM roles

| Model | Used for | API |
|---|---|---|
| `llama-3.1-8b-instant` | Question classification, router, cluster label generation | Groq (fast) |
| `llama-3.3-70b-versatile` | Wiki maintainer (tool calling), query synthesiser | Groq (capable) |
| `nomic-embed-text-v1.5` | Document and query embeddings | Local or Together AI |

## Environment variables

See `.env.example` for the full list. Required keys:

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase publishable key |
| `SUPABASE_SERVICE_KEY` | Supabase secret key (bypasses RLS) |
| `GROQ_API_KEY` | Groq API key |
| `EMBEDDING_PROVIDER` | `local` (default) or `together` |
| `SUPABASE_DB_URL` | Direct PostgreSQL connection (for migrations only) |
