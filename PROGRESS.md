# Build Progress

## Completed this session — Steps 1–5 of the build order

### Step 1 — Database schema ✅
All 8 migration files written in `backend/migrations/`:
- `001_extensions.sql` — pgvector
- `002_core_tables.sql` — surveys, survey_questions, survey_responses, open_ended_answers
- `003_cluster_tables.sql` — response_clusters
- `004_wiki_tables.sql` — wiki_pages, wiki_log (with updated_at trigger)
- `005_job_queue.sql` — ingestion_jobs, insight_cache
- `006_indexes.sql` — all HNSW and btree indexes
- `007_phase2_tables.sql` — events, participants (future use)
- `008_match_functions.sql` — match_wiki_pages, match_open_ended_answers, match_response_clusters (pgvector RPC functions)

**Action required:** Apply migrations to Supabase before first use. Two options:
1. Get DB password from Supabase Dashboard → Project Settings → Database → Connection string. Add `SUPABASE_DB_URL` to `.env`, then run `python scripts/apply_migrations.py`
2. Copy each `.sql` file into the Supabase Dashboard SQL Editor and run in order.

### Step 2 — Ingestion API ✅
- `POST /api/ingest/upload` — parses file, writes surveys/questions/responses, creates job row
- `GET /api/ingest/status/{jobId}` — polls job progress
- Handles CSV and XLSX; detects SurveyMonkey 2-row header format automatically
- Returns immediately after Stage 3 (parse/store); Stages 4–6 run in the background worker
- 50 MB file limit; validates extension before processing

### Step 3 — Dataset API ✅
- `GET /api/datasets` — lists all surveys with processing status
- `GET /api/datasets/{id}` — single survey
- `DELETE /api/datasets/{id}` — cascade-deletes all data

### Step 4 — Chat API (stub) ✅
- `POST /api/chat` — SSE streaming response; query agent wired up
- `GET /api/wiki` / `GET /api/wiki/{slug}` — browse wiki knowledge base
- Session storage (in-memory for MVP; PostgreSQL persistence is Step 10)

### Step 5 — All core services ✅
- **Parser** (`services/parser.py`): standard + SurveyMonkey formats; 26 unit tests
- **Classifier** (`services/classifier.py`): Groq llama-3.1-8b batch classification; 10 unit tests
- **Embedder** (`services/embedder.py`): nomic-embed-text-v1.5 (local or Together AI)
- **Clusterer** (`services/clusterer.py`): HDBSCAN + adaptive min_cluster_size + LLM labelling
- **Wiki Maintainer** (`services/wiki_maintainer.py`): LLMWiki ingest/lint via Groq tool calling
- **Query Agent** (`services/query_agent.py`): parallel wiki+SQL+semantic retrieval + synthesis
- **Worker** (`worker/pipeline.py`): persistent job queue; all 6 stages wired

### Test results ✅
```
48 passed, 0 failed
```
- 26 parser unit tests (format detection, SurveyMonkey parsing, error cases)
- 10 classifier unit tests (JSON parsing, nan handling, Groq mock)
- 12 API endpoint tests (upload, status, datasets, wiki)

---

## Next session — Steps 11–12

### Step 6 — Apply migrations and verify Supabase connection ✅
- `SUPABASE_DB_URL` added to root `.env` (session pooler, eu-west-1, port 5432 with sslmode=require)
- `backend/app/config.py` updated to load `.env` from repo root via absolute path (`Path(__file__).parent.parent.parent / ".env"`) — no more `backend/.env` needed
- `backend/.env` deleted; single `.env` in repo root now used by everything
- All 8 migrations applied successfully: extensions, core tables, cluster tables, wiki tables, job queue, indexes, phase2 tables, match functions

### Step 7 — Test real ingestion end-to-end ✅
- Fixed `database.py`: replaced broken `lru_cache` + `run_until_complete` with a lazy async singleton (double-checked lock pattern). 48/48 tests remain green.
- Fixed `pipeline.py`: "store" stage was silently skipped by the worker (only "parse" was whitelisted as a pass-through). Added "store" to the pass-through list so jobs advance correctly.
- Uploaded `UK Brand Survey Blank Data - 15.5.26.csv` (1000 responses, 110 questions). Full 6-stage pipeline completed successfully.
- Added `einops` to `requirements.txt` (required by nomic-embed-text-v1.5 but was missing).
- Fixed `pipeline.py` embed stage: switched from 1000 individual DB updates to bulk upserts in batches of 100 (10× faster).
- Fixed `pipeline.py` cluster stage: pgvector values come back from Supabase as JSON strings — added `_parse_vec()` helper to convert them to float lists before passing to numpy.
- Fixed `pipeline.py` cluster stage: switched from per-answer individual updates to batched upserts for `theme_cluster` writes.

### Step 8 — Verify wiki maintainer with real survey ✅
- Fixed `wiki_maintainer.py`: `msg.content` can be `None` when Groq returns only tool calls — added `or ""` guard.
- Fixed `wiki_maintainer.py`: tool response messages must be separate per `tool_call_id`, not batched in one JSON blob.
- 4 wiki pages created from the UK Brand Survey: `survey/uk-brand-survey-2026`, `entity/parkrun-participants`, `theme/sponsorship-importance`, `theme/parkrun-community-attitudes`.

### Step 9 — Supabase RPC functions ✅
- All three RPC functions smoke-tested via `scripts/test_rpc.py`:
  - `match_wiki_pages` → 3 rows returned ✓
  - `match_open_ended_answers` → 3 rows returned ✓
  - `match_response_clusters` → 3 rows returned (label="Strong Agreement on Parkrun" n=97, label="Sponsorship Importance" n=21) ✓
- Note: pgvector similarity values come back as strings from Supabase REST; any code reading them must call `float()` before arithmetic.

### Step 10 — Frontend integration ✅
- `frontend/.env.local`: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`, `NEXT_PUBLIC_MOCK_MODE=false`
- `frontend/services/upload.service.ts`: real multipart upload → polls `/api/ingest/status/{jobId}` → fetches completed dataset record. Mock path retained behind `mockMode` flag.
- `frontend/services/datasets.service.ts`: real `GET/DELETE /api/datasets` via `apiFetch`, mock path retained.
- `frontend/services/chat.service.ts`: real SSE stream via `parseSSE()` async generator; handles `chunk`, `sources`, `error`, `done` events. Mock path retained.
- `frontend/store/chatStore.ts`: updated `sendMessage` to use `onChunk` callback for real-time token streaming into the assistant bubble.
- `frontend/services/api.ts`: `apiFetch` now handles 204 No Content (DELETE) without crashing on `res.json()`.
- TypeScript: `npx tsc --noEmit` passes with zero errors.

### Step 11 — Insight cache + wiki lint cron ✅
- **Insight cache** added to `query_agent.py`: SHA-256 keyed on `question + mode + dataset_ids`. Standard-mode queries check `insight_cache` before any LLM/DB calls, and write results back with a 24-hour TTL. Deep-research queries bypass the cache (too user-specific).
- **Nightly lint cron** added to `pipeline.py`: `_nightly_lint_loop()` runs as a background asyncio task started by `run_worker()`. Wakes at midnight UTC each day, calls `wiki_maintainer.run_lint(db)`, logs the markdown report.
- 48/48 tests still green.

### Step 12 — Production deployment ✅ (infrastructure ready; deploy manually)
Configuration is in place — deployment itself requires manual steps in Railway and Vercel dashboards.

**What's been done:**
- `backend/railway.toml` — Nixpacks build + start command for the API service. Create a second Railway service with start command `python -m app.worker.pipeline`.
- `backend/Procfile` — alternative `web` + `worker` process definitions.
- `frontend/vercel.json` — Vercel config with security headers.
- `backend/migrations/009_rls_policies.sql` — RLS policies for all tables. Service role gets full access; authenticated users get read access; anon role gets nothing.

**To deploy:**

1. **Supabase RLS** — Run migration 009 via `python scripts/apply_migrations.py` (or the SQL Editor).

2. **Railway (backend)**
   - Create a new Railway project, connect the GitHub repo.
   - Add service 1 (API): root directory = `backend`, start command = `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add service 2 (worker): root directory = `backend`, start command = `python -m app.worker.pipeline`
   - Set environment variables (copy from `.env`): `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY`, `EMBEDDING_PROVIDER=local`, `CORS_ORIGINS=https://your-vercel-url.vercel.app`
   - Note: `SUPABASE_DB_URL` is only needed for migrations, not for the running app.
   - Note: local sentence-transformers model will be downloaded on first Railway deploy (~300 MB). Consider switching `EMBEDDING_PROVIDER=together` and setting `TOGETHER_API_KEY` for faster cold starts.

3. **Vercel (frontend)**
   - Import the GitHub repo in Vercel, set root directory = `frontend`.
   - Add environment variables: `NEXT_PUBLIC_API_BASE_URL=https://your-railway-api-url.up.railway.app`, `NEXT_PUBLIC_MOCK_MODE=false`

**Blockers / known gaps before production is ready:**
- Chat session persistence is in-memory — sessions are lost on API restart. A `chat_sessions` table and Supabase persistence would be needed before multi-user production use.
- No authentication layer — all data is publicly readable via the API. RLS migration 009 guards direct DB access, but the API endpoints themselves have no auth. Add Supabase Auth + JWT verification as a follow-up.
- Local embedding model (`nomic-embed-text-v1.5`, ~300 MB) requires a Railway plan with sufficient disk and RAM, or switch to `EMBEDDING_PROVIDER=together`.

---

## Remaining gaps before production

1. **Chat session persistence**: Sessions are in-memory — lost on API restart. Add a `chat_sessions` table and persist in Supabase before multi-user use.

2. **API authentication**: No auth on API endpoints. Add Supabase Auth + JWT verification. RLS migration 009 blocks direct DB access; the API layer is still open.

3. **Real token streaming**: Chat currently streams word-by-word with a 10 ms delay. Wire `stream=True` to Groq's async generator for true per-token streaming.

4. **Production embedding**: Local `nomic-embed-text-v1.5` requires ~300 MB disk + RAM on Railway. Set `EMBEDDING_PROVIDER=together` with `TOGETHER_API_KEY` for faster cold starts in production.
