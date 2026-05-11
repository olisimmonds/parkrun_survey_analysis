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

## Next session — Steps 6–12

### Step 6 — Apply migrations and verify Supabase connection
**Blocker:** Need the Supabase database password to apply migrations.
1. Go to Supabase Dashboard → Project Settings → Database
2. Copy the "Direct connection" URI (not the pooler)
3. Add to `.env` as `SUPABASE_DB_URL=postgresql://...`
4. Run: `python scripts/apply_migrations.py`
5. Verify tables are created in the Supabase Table Editor

### Step 7 — Test real ingestion end-to-end
With migrations applied:
1. Start the API: `uvicorn app.main:app --reload --port 8000`
2. Start the worker: `python -m app.worker.pipeline`
3. Upload `data/surveys/mock_survey_data.xlsx` via `POST /api/ingest/upload`
4. Watch job progress via `GET /api/ingest/status/{jobId}`
5. Verify wiki pages appear in `wiki_pages` table after Stage 6

**Known issue to resolve:** The supabase-py async client creation in `database.py` uses `asyncio.get_event_loop().run_until_complete()` which may not work in all async contexts. May need to switch to lazy initialisation using a startup lifespan event.

### Step 8 — Verify wiki maintainer with real survey
After Step 7 works:
1. Check `wiki_pages` table for generated pages
2. Test lint via direct Python call: `await wiki_maintainer.run_lint(db)`
3. Tune `wiki/SCHEMA.md` based on output quality

### Step 9 — Supabase RPC functions
The match functions in `008_match_functions.sql` need to be tested:
```python
result = await db.rpc("match_wiki_pages", {"query_embedding": [...], "match_count": 5}).execute()
```
Verify they return results via the Supabase Dashboard's API testing tab.

### Step 10 — Frontend integration
Update `frontend/services/` to call the real API instead of mocks:
- `upload.service.ts` → `POST /api/ingest/upload` + poll `/api/ingest/status/{jobId}`
- `datasets.service.ts` → `GET/DELETE /api/datasets`
- `chat.service.ts` → `POST /api/chat` (SSE stream handling)

The frontend already has `NEXT_PUBLIC_MOCK_MODE=true` in `.env.local`. Set to `false` and add `NEXT_PUBLIC_API_URL=http://localhost:8000`.

### Step 11 — Insight cache + wiki lint cron
- Add a nightly lint job: call `wiki_maintainer.run_lint(db)` on a schedule
- Add cache key hashing to `query_agent.py` for expensive map-reduce results
- Implement cache lookup before running deep-research queries

### Step 12 — Production deployment
- Deploy FastAPI + worker to Railway (two processes: `web` + `worker`)
- Deploy frontend to Vercel
- Set environment variables in Railway and Vercel dashboards
- Enable Supabase RLS policies for multi-user access

---

## Outstanding design questions

1. **Supabase async client init**: The current `database.py` uses `run_until_complete` which is fragile. Should use the lifespan pattern to initialise once at startup and store on `app.state`.

2. **Worker process separation**: For production, the worker and API are separate Railway services. For local dev, run both manually. Consider adding a `--with-worker` flag to `main.py` that starts the worker as a background asyncio task.

3. **Groq tool calling stability**: The wiki maintainer uses multi-turn tool calling with `llama-3.3-70b-versatile`. Test with a real survey before relying on this in production — Groq may throttle long tool-calling sessions. Fallback: Claude Haiku.

4. **SSE streaming**: Currently the chat endpoint streams words with a 10ms delay. Real token streaming from Groq requires `stream=True` and consuming the async generator. Add this in Step 10.

5. **Session persistence**: Chat sessions are currently in-memory (`_sessions` dict in `chat.py`). Add a `chat_sessions` table to Supabase and persist sessions there for the production build.
