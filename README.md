## About this project

This tool was built by a [parkrun Digital Ambassador](https://www.parkrun.com) volunteer
to explore how AI and data science can help parkrun generate better insights from their
large volume of community surveys.

parkrun is a free, community event where you can walk, jog, run, volunteer or spectate.
parkrun is 5k and takes place every Saturday morning. junior parkrun is 2k, dedicated to
4-14 year olds and their families, every Sunday morning. parkrun is positive, welcoming
and inclusive, there is no time limit and no one finishes last. Everyone is welcome to
come along.

---

## What this tool does

Upload a survey export (CSV or XLSX) and ask questions about your data in plain English.

The system builds a compounding knowledge base — not just a search engine. Each new
survey is processed into structured wiki pages (themes, trends, contradictions) that
persist and grow richer with every upload. The AI answers questions by reading the
knowledge base first, then pulling in raw numbers and verbatim quotes as supporting
evidence.

---

## Architecture

Three processing layers feed each other:

```
Raw CSV/XLSX
    │  parse + store
    ▼
SQL tables + vector embeddings (Supabase + pgvector)
    │  HDBSCAN clustering + LLM labelling
    ▼
Structured theme clusters
    │  wiki maintainer (Groq tool-calling)
    ▼
Wiki layer — interlinked pages: themes, entities, surveys, trends
    │  primary knowledge source at query time
    ▼
Query agent (wiki + SQL + semantic search) → streamed answer
```

The wiki compounds: ingesting a second survey on the same topic updates existing pages,
flags contradictions with earlier findings, and links related themes — work that would
otherwise happen manually in a spreadsheet.

---

## What happens when you submit data

### Survey CSV or XLSX

1. **Upload** — The file is sent to the backend API. CSV and XLSX are supported;
   SurveyMonkey's two-row header format is detected automatically.

2. **Parse** — Questions are extracted from the column headers. Each respondent's
   answers are stored as a structured row in the database.

3. **Classify** — An AI (Groq llama-3.1-8b) reads each question label and a sample of
   its answers to classify it: `rating`, `multiple_choice`, `open_ended`, or
   `demographic`. This determines how each column is indexed and queried later.

4. **Embed** — Every open-ended text answer is converted into a 768-dimensional vector
   by the `nomic-embed-text-v1.5` model running locally. These vectors capture semantic
   meaning, not just keywords.

5. **Cluster** — HDBSCAN groups answers to the same question by similarity. The number
   of clusters adapts to response volume. An AI (Groq llama-3.1-8b) labels each cluster
   with a concise theme name and a 2–3 sentence summary.

6. **Wiki update** — A more capable AI (Groq llama-3.3-70b) reads all cluster summaries
   and writes or updates wiki pages. A survey on volunteer experience might produce pages
   like *Theme: Motivation for Volunteering*, *Theme: Barriers to Continuing*, and
   *Survey: UK Volunteer Experience Survey 2026*. If a conflicting theme appeared in an
   earlier survey, a *Contradiction* page is created automatically.

Once complete (typically 5–15 minutes for a thousand-row survey), the dataset appears
as **ready** in the Data page and all wiki pages are searchable from the Chat page.

### PDF file

Not supported. The upload endpoint accepts only `.csv`, `.xlsx`, and `.xls`. Submitting
a PDF returns a validation error immediately.

### Numerical or operational data (CSV without open-ended text)

The same pipeline runs. Numerical columns are classified as `rating` or
`multiple_choice` and stored in the database, making them available for SQL-based
queries ("what was the average satisfaction score?"). Clustering and wiki-page creation
only apply to open-ended text columns — a purely numerical file will ingest successfully
but may produce no wiki pages. It will still be queryable from the Chat page for
aggregations and statistics.

---

## Getting started

**Prerequisites:** Python 3.11+, Node.js 18+, a Supabase project (free tier is fine),
a Groq API key (free tier).

```bash
# 1 — clone and create .env
cp .env.example .env   # fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY, GROQ_API_KEY, SUPABASE_DB_URL

# 2 — apply database migrations
cd backend
pip install -r requirements.txt
python scripts/apply_migrations.py

# 3 — install frontend dependencies
cd ../frontend
npm install
```

---

## Running locally

Three processes, three terminals:

```bash
# Terminal 1 — API (http://localhost:8000)
cd backend && uvicorn app.main:app --reload --port 8000

# Terminal 2 — background pipeline worker
cd backend && python -m app.worker.pipeline

# Terminal 3 — frontend (http://localhost:3000)
cd frontend && npm run dev
```

Open **http://localhost:3000**. API docs at **http://localhost:8000/api/docs**.

The worker must be running for uploads to process beyond the initial parse step. The first upload after a fresh worker start takes ~45 extra seconds while the embedding model loads.

---

## Running a demo (sharing with others)

The frontend is deployed on Vercel. The backend runs on your machine and is exposed via a Cloudflare tunnel.

**Before the demo:**

```powershell
# From the repo root — downloads cloudflared automatically if needed
.\start-demo.ps1
```

The script starts the API, worker, and tunnel, then prints a URL like `https://abc123.trycloudflare.com`.

Go to **Vercel → your project → Settings → Environment Variables**, set `NEXT_PUBLIC_API_BASE_URL` to that URL, and click **Redeploy** (~30 seconds). Your Vercel URL is then shareable.

**The tunnel URL changes each session** — update the Vercel env var before each demo.

---

## Next steps

Items already discussed and scoped for future work:

- **Real-time token streaming** — Wire Groq's `stream=True` to the chat endpoint so
  answers appear token-by-token rather than word-by-word.
- **Chat session persistence** — Store sessions in Supabase so conversations survive
  API restarts.
- **API authentication** — Add Supabase Auth + JWT verification to all API endpoints.
  The RLS migration (`009_rls_policies.sql`) is ready; the application layer is not yet
  gated.
- **Production deployment** — Railway (API + worker) + Vercel (frontend). Config files
  are in place; see `PROGRESS.md` Step 12 for exact steps.
- **Phase 2 data sources** — Event attendance, finish times, volunteer rosters. The
  database schema already has the tables; the ingestion pipeline handles CSV out of the
  box.
- **Together AI embeddings** — Switch `EMBEDDING_PROVIDER=together` to offload the
  300 MB local model to a hosted API, reducing Railway memory requirements.

---

## Contributing

Contributions are welcome. To report a bug or suggest a feature, open an issue on GitHub.

---

## Privacy and data

- All survey data uploaded through this tool is stored in the **Supabase project you
  configure** — data leaves your machine and is held in Supabase's cloud database.
  Review [Supabase's privacy policy](https://supabase.com/privacy) before uploading
  sensitive data.
- When generating insights, **text from survey responses** (open-ended answers and
  cluster samples) is sent to **Groq's API**. Review [Groq's privacy
  policy](https://groq.com/privacy-policy/) before use.
- API keys (`GROQ_API_KEY`, Supabase keys) are stored in your local `.env` file and
  are never committed to version control (`.env` is in `.gitignore`).
