# Backend Architecture Plan — parkrun Insights Platform

> **Scope of this document:** MVP focuses on survey ingestion and AI-powered querying at scale. The architecture is designed so that operational data (event attendance, participation records) can be added as a second data source later without redesigning the core system.

---

## 1. The Core Problem

parkrun generates a vast corpus of community data:

- **Hundreds of surveys** covering volunteers, participants, Run Directors, newcomers, wellbeing, etc.
- **Potentially millions of individual responses**, each containing a mix of structured answers (ratings, checkboxes, demographics) and unstructured text (open-ended questions)
- Questions that require insight *across the entire corpus* — not just finding relevant snippets, but synthesising patterns, trends, and meta-themes

Simple Retrieval-Augmented Generation (RAG) is not sufficient here. RAG is designed to fetch a handful of relevant text chunks and feed them to an LLM as context. If a user asks *"How has volunteer satisfaction changed over the past three years, and what are the recurring themes in their open-ended feedback?"*, a RAG system would:

- Only retrieve a small, arbitrarily sampled slice of responses
- Miss statistical aggregations entirely (trend analysis, percentage breakdowns)
- Fail to synthesise themes that emerge from thousands of individual data points

The system needs to operate at two levels simultaneously:
1. **Quantitative layer** — SQL-queryable structured data for aggregations, trends, and statistics
2. **Qualitative layer** — semantic search and clustering over embedding-indexed open-ended responses

An **agentic framework** bridges these: an orchestrating LLM decides which tools to invoke (SQL query, semantic cluster search, summarisation), chains the results, and synthesises a coherent narrative answer.

---

## 2. Data Sources (MVP and beyond)

### MVP
| Source | Format | Volume estimate |
|---|---|---|
| SurveyMonkey / Google Forms exports | CSV (2-row headers) | Hundreds of files, up to ~5,000 rows each |
| Volunteer experience surveys | CSV | ~2,000 rows/year |
| Participant wellbeing surveys | CSV | ~5,000 rows/quarter |
| Run Director feedback | CSV | ~500 rows/survey |

### Phase 2 (extensible model)
| Source | Format | Notes |
|---|---|---|
| Event attendance | CSV / API | 50,000+ rows/year across UK events |
| Participant finish times | CSV / API | Operational, high volume |
| Volunteer rosters | CSV | Per-event role allocations |
| Demographics / diversity data | CSV | Linked to survey responses |

The data model is designed to accommodate Phase 2 without schema migration of the core survey tables.

---

## 3. Storage: Supabase

**Why Supabase:**
- PostgreSQL at its core — mature, reliable, excellent for structured survey data
- `pgvector` extension built-in — stores and queries embeddings without a separate vector DB
- Supabase Storage — handles raw uploaded files (CSV, PDF, XLSX)
- Row-level security — can scope data access per user/organisation in future
- Edge Functions (Deno) — for lightweight ingestion hooks
- Generous free tier → scales linearly; avoids over-engineering early

**Why not a dedicated vector DB (Pinecone, Weaviate):**
- Adds operational complexity for marginal performance gain at this scale
- pgvector with HNSW indexing handles millions of vectors well
- Keeps everything in one place, simpler joins between structured + embedding data

---

## 4. Data Model

### Core Tables

```sql
-- Survey catalogue
CREATE TABLE surveys (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  type        TEXT NOT NULL,          -- 'volunteer', 'participant', 'run_director', etc.
  source      TEXT,                   -- 'surveymonkey', 'google_forms', 'manual'
  conducted_at TIMESTAMPTZ,           -- when the survey was fielded
  uploaded_at TIMESTAMPTZ DEFAULT now(),
  uploaded_by UUID REFERENCES auth.users,
  row_count   INT,
  metadata    JSONB                   -- arbitrary extra fields, extensible
);

-- Questions extracted from survey headers
CREATE TABLE survey_questions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id   UUID REFERENCES surveys ON DELETE CASCADE,
  column_key  TEXT NOT NULL,          -- raw CSV column name
  label       TEXT NOT NULL,          -- cleaned question text
  question_type TEXT NOT NULL,        -- 'rating', 'multiple_choice', 'open_ended', 'demographic'
  position    INT,
  options     JSONB                   -- for MC questions: list of valid answers
);

-- Individual survey responses (one row = one participant's full response)
CREATE TABLE survey_responses (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id   UUID REFERENCES surveys ON DELETE CASCADE,
  respondent_ref TEXT,               -- anonymised identifier if present
  responded_at TIMESTAMPTZ,
  structured  JSONB NOT NULL         -- { question_id: value, ... } for all structured answers
);

-- Open-ended text answers with embeddings (separate for performance)
CREATE TABLE open_ended_answers (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  response_id UUID REFERENCES survey_responses ON DELETE CASCADE,
  question_id UUID REFERENCES survey_questions ON DELETE CASCADE,
  answer_text TEXT NOT NULL,
  embedding   VECTOR(768),           -- populated async after ingestion
  theme_cluster INT                  -- populated by clustering job
);

-- Pre-computed theme clusters (generated by background job)
CREATE TABLE response_clusters (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id   UUID REFERENCES surveys,
  question_id UUID REFERENCES survey_questions,
  cluster_id  INT NOT NULL,
  label       TEXT,                  -- LLM-generated theme label
  summary     TEXT,                  -- LLM-generated cluster summary
  response_count INT,
  representative_quotes JSONB,       -- sample verbatim quotes
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Cached insights (avoid re-running expensive analysis)
CREATE TABLE insight_cache (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key   TEXT UNIQUE NOT NULL,  -- hash of query + dataset scope
  result      JSONB NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now(),
  expires_at  TIMESTAMPTZ
);
```

### Phase 2 Extension (non-breaking additions)

```sql
-- Operational data — separate schema, joinable via event/date
CREATE TABLE events (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_name  TEXT NOT NULL,
  location    TEXT,
  event_date  DATE NOT NULL,
  finishers   INT,
  volunteers  INT,
  first_timers INT,
  metadata    JSONB
);

-- Participation records — links to surveys via respondent_ref
CREATE TABLE participants (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  athlete_ref     TEXT UNIQUE,       -- anonymised parkrun ID
  home_event      TEXT,
  total_runs      INT,
  first_run_date  DATE,
  metadata        JSONB
);
```

### Key Indexes

```sql
-- Vector similarity search (HNSW — fast approximate nearest neighbour)
CREATE INDEX ON open_ended_answers USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Structured query performance
CREATE INDEX ON survey_responses (survey_id);
CREATE INDEX ON open_ended_answers (question_id, theme_cluster);
CREATE INDEX ON surveys (type, conducted_at);
```

---

## 5. Ingestion Pipeline

When a user uploads a CSV, the following steps run:

```
Upload → Parse → Classify → Store Structured → Embed → Cluster
```

### Step 1: Parse
- Detect file type (CSV, XLSX, PDF)
- For CSV: handle SurveyMonkey's 2-row header format (forward-fill column names)
- Extract metadata: row count, column names, date fields
- Run in a Next.js API Route or Supabase Edge Function

### Step 2: Classify Questions
- Use an LLM (Mistral Small — fast and cheap) to classify each column as:
  - `rating` — numeric scale (1–5, 1–10, NPS)
  - `multiple_choice` — bounded set of options
  - `open_ended` — free text requiring embedding
  - `demographic` — age, gender, location, frequency
  - `datetime` — response timestamps
- This classification drives downstream storage and analysis strategy

### Step 3: Store Structured Data
- Write one `survey_response` row per CSV row
- Pack all structured answers into the `structured` JSONB column
- Fast, synchronous — user sees confirmation immediately

### Step 4: Embed Open-Ended Responses (async)
- For every `open_ended` column, extract all non-empty answers
- Batch embed using **nomic-embed-text** (open-source, 768-dim, runs via Ollama or Together AI)
  - Alternative: OpenAI `text-embedding-3-small` if cost is acceptable
- Write embeddings to `open_ended_answers.embedding`
- Run as a background job (Supabase Edge Function triggered by DB insert, or a Python worker)

### Step 5: Cluster (background job, runs after embedding)
- For each question, cluster the embeddings using **HDBSCAN** (better than K-means for varying cluster sizes)
- For each cluster: call Mistral to generate a theme label and summary
- Populate `response_clusters` table
- This is the key enabler for meta-insight queries

### Ingestion Rate Considerations
- A 5,000-row survey with 3 open-ended questions = 15,000 embeddings
- At ~2,000 embeddings/second (nomic via API): ~8 seconds
- Clustering and summarisation: ~30–60 seconds per question
- All async — user doesn't wait; UI shows "processing" state

---

## 6. Query Architecture: The Analyst Agent

The agent is an orchestrating LLM (Mistral Large or Claude 3.5 Sonnet) that receives a user question, decides which tools to call, chains results, and produces a synthesis.

### Tool Inventory

| Tool | Description | Backed by |
|---|---|---|
| `sql_query` | Executes validated SQL against survey + structured data | Supabase PostgreSQL |
| `semantic_search` | Finds top-K relevant open-ended answers via pgvector | pgvector HNSW index |
| `cluster_summary` | Returns pre-computed theme clusters for a question/survey | `response_clusters` table |
| `trend_analysis` | Compares metric across surveys by date range | SQL + aggregation |
| `cross_survey_themes` | Finds themes present across multiple surveys | Cluster intersection |
| `quote_sampler` | Pulls representative verbatim quotes for a theme | `open_ended_answers` |

### Agent Flow

```
User question
    ↓
[Router LLM — Mistral Small] ← fast, cheap classification
  Classifies: Quantitative / Qualitative / Mixed
    ↓
[Planner LLM — Mistral Large]
  Produces a tool-call plan: which tools, in what order, with what params
    ↓
[Tool Execution — parallel where possible]
  sql_query → numerical results
  cluster_summary → theme data
  semantic_search → supporting quotes
    ↓
[Synthesiser LLM — Claude 3.5 Sonnet or Mistral Large]
  Receives all tool outputs
  Writes the final narrative response with citations
    ↓
User response
```

### Why Three LLMs

| LLM | Role | Rationale |
|---|---|---|
| **Mistral Small** (open-source) | Routing, classification, question parsing, SQL generation | Fast (sub-second), cheap or free, good at structured tasks |
| **Mistral Large** (or Mixtral 8x22B) | Planning tool-call sequences, cluster summarisation | Strongest open-source reasoning; avoids proprietary API cost for most work |
| **Claude 3.5 Sonnet / GPT-4o** | Final synthesis and narrative | Best-in-class for nuanced, long-form writing; used sparingly (once per query) |

This layered approach keeps costs low while using the best model only for the output the user actually sees.

---

## 7. Handling Scale: Beyond Simple RAG

The key insight is that we pre-process the corpus into queryable structures so the agent doesn't need to read millions of rows at query time.

| Challenge | Solution |
|---|---|
| 1M+ open-ended responses in context | **Cluster summaries** — agent reads cluster labels/summaries (hundreds of tokens) not raw responses |
| Cross-survey theme detection | **Cross-survey cluster intersection** — find clusters with cosine similarity > threshold across surveys |
| Year-over-year trend analysis | **SQL aggregations** over structured JSONB — no LLM needed for the numbers |
| Finding representative quotes | **pgvector search within a cluster** — semantically close to cluster centroid |
| First-time insights on new surveys | **On-demand clustering** — triggered at upload, cached in `response_clusters` |
| Repeated identical queries | **Insight cache** — hash of query + dataset scope, 24hr TTL |

### Map-Reduce for Novel Queries
When a user asks something that can't be served from the cache or clusters:

1. **Map:** Split open-ended responses into batches of ~100; call Mistral Small to extract relevant themes from each batch in parallel
2. **Reduce:** Call Mistral Large to synthesise the batch summaries into a unified answer
3. **Cache** the result for future identical queries

This allows full-corpus analysis without hitting context window limits.

---

## 8. API Layer Design

```
POST /api/ingest/upload          ← receives file, starts ingestion pipeline
GET  /api/ingest/status/:jobId   ← polls ingestion progress
GET  /api/datasets               ← list all datasets with metadata
DELETE /api/datasets/:id         ← remove dataset + cascade delete embeddings
POST /api/chat                   ← send user message, returns streaming response
GET  /api/chat/sessions          ← list past conversations
```

All endpoints authenticated via Supabase Auth (JWT). Row-level security on the DB ensures users only see their own data.

---

## 9. Tech Stack Summary

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Next.js 16, React 19, TypeScript | Already built |
| Database | Supabase (PostgreSQL 15+) | Structured data + RLS |
| Vector store | pgvector (HNSW index) | Co-located with SQL |
| File storage | Supabase Storage | Raw CSVs, PDFs |
| Embedding model | nomic-embed-text (open-source) | Via Together AI or self-hosted |
| Clustering | HDBSCAN (Python scikit-learn-extra) | Background worker |
| Agent orchestration | Python (LangGraph or custom async) | Runs as a FastAPI service |
| LLM: Routing | Mistral Small (`mistral-small-latest`) | ~$0.001/1k tokens |
| LLM: Reasoning | Mistral Large 2 (`mistral-large-latest`) | ~$0.003/1k tokens |
| LLM: Synthesis | Claude 3.5 Sonnet or GPT-4o | Sparingly, final output only |
| Hosting (API) | Railway / Render / Fly.io | Simple Python FastAPI |
| Hosting (Frontend) | Vercel | Free tier ample |

---

## 10. Recommended Build Order

1. **Supabase setup** — create project, run schema migrations, enable pgvector
2. **Ingestion API** — upload endpoint, CSV parser, structured storage
3. **Embedding pipeline** — async job, nomic-embed-text, write to pgvector
4. **Basic SQL agent** — natural language → SQL → results (handles all quantitative questions)
5. **Cluster pipeline** — HDBSCAN, Mistral cluster labelling, store summaries
6. **Semantic search agent** — pgvector search + quote retrieval
7. **Orchestrating agent** — planner + tool execution + synthesiser
8. **Frontend integration** — swap mock services for real API calls
9. **Caching layer** — insight cache for repeated queries
10. **Phase 2 data sources** — operational/participation tables

Steps 1–4 deliver a working MVP. Steps 5–7 unlock the deep qualitative meta-insight capability.

---

## 11. Open Questions for Decision

- **Embedding model hosting:** Together AI API (easiest) vs. self-hosted nomic via Ollama (cheapest at scale) vs. OpenAI (best ecosystem but proprietary)?
- **Agent framework:** LangGraph (structured, good for complex multi-step agents) vs. custom async Python (simpler, fewer dependencies)?
- **Authentication:** Supabase Auth (built-in) vs. integrate with an existing parkrun identity system?
- **Multi-tenancy:** Should different parkrun regions/teams have isolated data, or is this a shared corpus?
- **Deep Research mode:** In the UI, this mode already exists — in the backend, this should trigger the full map-reduce pipeline rather than the fast cluster-summary path.
