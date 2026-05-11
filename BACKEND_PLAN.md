# Backend Architecture Plan — parkrun Insights Platform

> **Scope:** MVP focuses on survey ingestion and AI-powered querying at scale, with the architecture extensible to operational and participation data in Phase 2. This document describes a unified system — not a collection of independent ideas — where every component feeds the next.

---

## 1. Core Premise

parkrun generates a large corpus of community data:

- Hundreds of surveys covering volunteers, participants, Run Directors, newcomers, and wellbeing
- Potentially millions of individual responses per year, mixing structured ratings with rich open-ended text
- Users who need meta-level insights — not "find me a quote", but "how has volunteer motivation shifted over three years, and what's driving it?"

**Simple RAG fails** because it retrieves a handful of text chunks on every query and re-derives understanding from scratch. It cannot aggregate, cannot trend-analyse, and misses themes that only emerge across thousands of responses read together.

**HDBSCAN clustering** is needed, but clusters alone are per-survey artefacts. Every query must re-examine cluster data; nothing compounds. Ask the same question twice and the system does the same work twice.

**What is needed is a knowledge layer that compounds.** Each time a survey is ingested, the system should grow *smarter* — not just store more rows. Cross-survey relationships should be discovered once and remembered. Contradictions should be flagged at ingest time. When a good answer is produced, it should become part of the corpus.

The architecture is three processing layers feeding each other, with the **Wiki layer** as the primary knowledge base that accumulates structured narrative knowledge like a living research document. The query agent reads the wiki first; SQL and raw vector search play supporting roles for numerical precision and verbatim quotes.

```
┌─────────────────────────────────────────────────────────┐
│                    RAW DATA LAYER                        │
│   Uploaded CSVs → SQL tables + pgvector embeddings       │
└───────────────────────┬─────────────────────────────────┘
                        │ background job pipeline
┌───────────────────────▼─────────────────────────────────┐
│               STRUCTURED KNOWLEDGE LAYER                 │
│   HDBSCAN clusters · cluster summaries · SQL aggregations│
└───────────────────────┬─────────────────────────────────┘
                        │ wiki maintainer (LLM + SCHEMA.md)
┌───────────────────────▼─────────────────────────────────┐
│                   WIKI LAYER (LLMWiki)                   │
│   Interlinked pages: themes, entities, surveys, trends,  │
│   contradictions · Compounds with every ingest           │
│   Stored as rows in wiki_pages (PostgreSQL)              │
└───────────────────────┬─────────────────────────────────┘
                        │ primary knowledge source at query time
              ┌─────────▼────────┐
              │   QUERY AGENT    │  SQL + semantic search as supporting tools
              └─────────┬────────┘
                        │
              ┌─────────▼────────┐
              │   USER RESPONSE  │
              └──────────────────┘
```

---

## 2. Data Sources

### MVP
| Source | Format | Volume estimate |
|---|---|---|
| SurveyMonkey / Google Forms exports | CSV (2-row headers) | Hundreds of files, up to ~5,000 rows each |
| Volunteer experience surveys | CSV | ~2,000 rows/year |
| Participant wellbeing surveys | CSV | ~5,000 rows/quarter |
| Run Director feedback | CSV | ~500 rows/survey |

### Phase 2 (extensible — no schema migration required)
| Source | Format | Notes |
|---|---|---|
| Event attendance | CSV / API | 50,000+ rows/year across UK events |
| Participant finish times | CSV / API | Operational, high volume |
| Volunteer rosters | CSV | Per-event role allocations |
| Demographics / diversity data | CSV | Linkable to survey `respondent_ref` |

Phase 2 data plugs into the same ingestion pipeline. The wiki layer gains new page types (Event pages, Participation trend pages) and the same clustering + wiki-update jobs run on the new data. The query agent gains new SQL tools but the architecture is unchanged.

---

## 3. Storage: Supabase

**Why Supabase:**
- PostgreSQL — reliable, excellent for structured survey data, supports JSONB, full-text search, and scheduled jobs natively
- `pgvector` extension — vector similarity search co-located with SQL, no separate vector DB needed
- Supabase Storage — raw file store for uploaded CSVs and PDFs (not wiki pages — see below)
- Row-level security — data can be scoped per user or organisation without application-layer guards
- Edge Functions (Deno) — lightweight webhook triggers for job queue notifications
- Free tier scales linearly; avoids premature infrastructure complexity

**Why not a dedicated vector DB (Pinecone, Weaviate):**
pgvector with HNSW indexing handles tens of millions of vectors well. The benefit of keeping vectors in the same database as structured data is that hybrid queries (vector similarity *plus* SQL filters like `WHERE survey.type = 'volunteer' AND conducted_at > '2023-01-01'`) are cheap single-database operations, not distributed joins.

**Wiki storage:** Wiki pages live entirely in the `wiki_pages` DB table — not in Supabase Storage. The LLMWiki pattern is a filesystem metaphor; we implement the same compounding knowledge concept with DB rows. Storing wiki pages in Storage as files AND syncing them to a DB table adds complexity with no benefit in a web-app context. The wiki_pages table is the wiki.

---

## 4. Data Model

### Core Tables

```sql
-- Survey catalogue
CREATE TABLE surveys (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name         TEXT NOT NULL,
  type         TEXT NOT NULL,          -- 'volunteer', 'participant', 'run_director', etc.
  source       TEXT,                   -- 'surveymonkey', 'google_forms', 'manual'
  conducted_at TIMESTAMPTZ,
  uploaded_at  TIMESTAMPTZ DEFAULT now(),
  uploaded_by  UUID REFERENCES auth.users,
  row_count    INT,
  metadata     JSONB
);

-- Questions extracted from survey headers
CREATE TABLE survey_questions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id     UUID REFERENCES surveys ON DELETE CASCADE,
  column_key    TEXT NOT NULL,
  label         TEXT NOT NULL,
  question_type TEXT NOT NULL,         -- 'rating', 'multiple_choice', 'open_ended', 'demographic'
  position      INT,
  options       JSONB
);

-- Individual participant responses
CREATE TABLE survey_responses (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id      UUID REFERENCES surveys ON DELETE CASCADE,
  respondent_ref TEXT,
  responded_at   TIMESTAMPTZ,
  structured     JSONB NOT NULL        -- { question_id: value } for all structured answers
);

-- Open-ended answers with embeddings
CREATE TABLE open_ended_answers (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  response_id   UUID REFERENCES survey_responses ON DELETE CASCADE,
  question_id   UUID REFERENCES survey_questions ON DELETE CASCADE,
  answer_text   TEXT NOT NULL,
  embedding     VECTOR(768),           -- populated async after ingestion
  theme_cluster INT                    -- populated by clustering job
);

-- Pre-computed HDBSCAN theme clusters
CREATE TABLE response_clusters (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id             UUID REFERENCES surveys,
  question_id           UUID REFERENCES survey_questions,
  cluster_id            INT NOT NULL,
  label                 TEXT,
  summary               TEXT,
  response_count        INT,
  centroid              VECTOR(768),   -- cluster centroid for cross-survey comparison
  representative_quotes JSONB,
  created_at            TIMESTAMPTZ DEFAULT now()
);
```

### Wiki Layer Tables

The wiki is implemented as Karpathy's LLMWiki pattern: a schema document teaches an LLM to maintain a set of interlinked, structured knowledge pages. In our case those pages are rows in `wiki_pages` rather than files on a filesystem — the compounding behaviour is identical.

```sql
-- One row per wiki page
CREATE TABLE wiki_pages (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug         TEXT UNIQUE NOT NULL,   -- e.g. 'theme/volunteer-motivation'
  page_type    TEXT NOT NULL,          -- 'theme', 'entity', 'survey', 'trend', 'contradiction', 'synthesis'
  title        TEXT NOT NULL,
  content      TEXT NOT NULL,          -- full markdown body
  embedding    VECTOR(768),            -- embedding of content for semantic wiki search
  linked_slugs TEXT[],                -- outbound [[wiki-links]] extracted from content
  survey_ids   UUID[],                -- surveys this page draws from
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Append-only log of all wiki operations
CREATE TABLE wiki_log (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type TEXT NOT NULL,            -- 'ingest', 'update', 'lint', 'filed_answer'
  page_slug  TEXT,
  survey_id  UUID REFERENCES surveys,
  summary    TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Persisted job queue for background pipeline (survives user leaving)
CREATE TABLE ingestion_jobs (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id    UUID REFERENCES surveys,
  stage        TEXT NOT NULL,          -- 'parse', 'embed', 'cluster', 'wiki_update', 'done'
  status       TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'done', 'failed'
  attempt      INT DEFAULT 0,
  last_error   TEXT,
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Insight cache
CREATE TABLE insight_cache (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key  TEXT UNIQUE NOT NULL,
  result     JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ
);
```

### Phase 2 Extension

```sql
CREATE TABLE events (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_name   TEXT NOT NULL,
  location     TEXT,
  event_date   DATE NOT NULL,
  finishers    INT,
  volunteers   INT,
  first_timers INT,
  metadata     JSONB
);

CREATE TABLE participants (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  athlete_ref    TEXT UNIQUE,
  home_event     TEXT,
  total_runs     INT,
  first_run_date DATE,
  metadata       JSONB
);
```

### Key Indexes

```sql
-- Vector similarity (HNSW — fast approximate nearest-neighbour)
CREATE INDEX ON open_ended_answers USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX ON response_clusters USING hnsw (centroid vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX ON wiki_pages USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Structured query performance
CREATE INDEX ON survey_responses (survey_id);
CREATE INDEX ON open_ended_answers (question_id, theme_cluster);
CREATE INDEX ON surveys (type, conducted_at);
CREATE INDEX ON wiki_pages (page_type);
CREATE INDEX ON ingestion_jobs (status, stage);
```

---

## 5. Resilient Ingestion Pipeline

### The Problem: Users Leave Mid-Process

A 5,000-row survey with several open-ended questions will take 2–5 minutes to fully process. Users must be able to close the browser immediately after confirming an upload and trust that all processing will complete correctly.

### Job Queue: Persistent State in PostgreSQL

Every upload creates a row in `ingestion_jobs` before any processing starts. The job queue is stored in the database — not in memory — so it survives server restarts, network interruptions, and tab closures. A Python worker polls `ingestion_jobs WHERE status = 'pending'` on a 10-second interval.

Each stage advances the job atomically:
```
pending:parse → running:parse → pending:embed → running:embed
  → pending:cluster → running:cluster → pending:wiki_update
  → running:wiki_update → done
```

Failed stages retry up to 3 times with exponential back-off before marking as `failed`.

### Stage-by-Stage Pipeline

**Stage 1 — Parse** (synchronous, ~1 second)
- Detect file type (CSV, XLSX)
- Handle SurveyMonkey's 2-row header format via forward-fill
- Write survey + question rows to DB
- Create `ingestion_jobs` row with `status=pending, stage=parse`
- User sees upload confirmation immediately

**Stage 2 — Classify Questions** (worker, ~5 seconds)
- Use Mistral Small (free tier) to classify each column: `rating`, `multiple_choice`, `open_ended`, `demographic`, `datetime`
- Classification drives all downstream decisions
- Update `survey_questions` with types

**Stage 3 — Store Structured Responses** (worker, ~10 seconds for 5k rows)
- Write one `survey_response` row per CSV row
- Pack all structured answers into `structured` JSONB
- Write raw open-ended text to `open_ended_answers` (embedding NULL for now)
- Advance job to `pending:embed`

**Stage 4 — Embed Open-Ended Answers** (worker, ~30–60 seconds)
- Batch all open-ended answers from this survey
- Call nomic-embed-text API in batches of 256
- Write 768-dimension vectors to `open_ended_answers.embedding`
- Advance job to `pending:cluster`

**Stage 5 — Cluster** (worker, ~60–120 seconds)
- Run HDBSCAN on the survey's embeddings per question
- Compute cluster centroids
- Call Mistral Small to label each cluster and write a 2–3 sentence summary
- Store `response_clusters` with centroid vectors
- Pick representative quotes (3 responses closest to each centroid)
- Advance job to `pending:wiki_update`

**Stage 6 — Wiki Update** (worker, ~60–120 seconds — see Section 6)
- Materialise cluster summaries, per-question statistics, and key quotes into a structured source document (markdown string)
- Call Mistral Small with `SCHEMA.md` + the wiki index (existing page slugs + one-line descriptions) + any relevant existing wiki pages + the new source document
- Mistral uses tool calling to write/update wiki pages
- Execute those tool calls: upsert rows to `wiki_pages`, recompute embeddings for changed pages, extract `linked_slugs`, append to `wiki_log`
- Advance job to `done`

### Ingestion Rate Estimates
| Survey size | Open-ended questions | Total time |
|---|---|---|
| 500 rows, 2 questions | 1,000 answers | ~1.5 min |
| 2,000 rows, 3 questions | 6,000 answers | ~3 min |
| 5,000 rows, 4 questions | 20,000 answers | ~6 min |

All fully async. The user can close the browser after Stage 1 completes.

---

## 6. The Wiki Layer: Karpathy's LLMWiki Pattern

### What the Pattern Is

Karpathy's LLMWiki is a schema document (a `SCHEMA.md` or `AGENTS.md` file) that you give to an LLM alongside source material. The LLM follows the schema to create and maintain a set of interlinked, structured knowledge pages — with cross-references, contradiction detection, trend tracking, and an index. No custom code handles any of this; the LLM follows the schema.

In the original gist, pages are markdown files on a filesystem. In this system, pages are rows in `wiki_pages`. The compounding behaviour — and the ingest/lint/query workflows — is identical.

`SCHEMA.md` lives in the codebase (version controlled), not in the database. It is read by the Python worker at runtime and included in every call to the wiki maintainer LLM.

### Page Types and Naming

| Type | Slug pattern | Purpose |
|---|---|---|
| `survey` | `survey/{slug}` | Key findings from a single survey |
| `theme` | `theme/{slug}` | Recurring theme across multiple surveys |
| `entity` | `entity/{slug}` | A group or concept referenced across surveys |
| `trend` | `trend/{slug}` | Time-series narrative for a metric |
| `contradiction` | `contradiction/{slug}` | Conflicting findings between surveys |
| `synthesis` | `synthesis/{slug}` | Filed-back answer from a user query |

### The Schema File (`SCHEMA.md`)

Written once by us, version-controlled in the codebase, read by Mistral on every wiki operation. This is the most important file in the system.

```markdown
# parkrun Insights Wiki — Schema

## Purpose
You are maintaining a compounding knowledge base built from parkrun community surveys.
Your role is disciplined wiki editor, not chatbot. Follow these conventions exactly.
Use the write_wiki_page tool to create or update pages. Do not narrate; act.

## Page types and naming
- survey/{slug}        Key findings from a single survey
- theme/{slug}         Recurring theme across multiple surveys
- entity/{slug}        A group or concept (volunteers, first-timers, Run Directors)
- trend/{slug}         Time-series narrative for a metric across years
- contradiction/{slug} Conflicting findings between two or more surveys
- synthesis/{slug}     Filed-back answer from a user query (do not create these)

## Ingest workflow
When given a new source document:
1. Read the source fully. Identify themes, statistics, entities, dates, and quotes.
2. Call write_wiki_page to create survey/{slug} summarising this survey's key findings.
3. For each theme found: if a matching theme/ page exists in the index, load it and
   update it with the new data and new survey reference. If no match exists, create one.
4. For each entity found: update or create entity/ pages accordingly.
5. Compare key statistics against related existing pages. If a metric differs by more
   than 5% from a prior year's page, call write_wiki_page for a contradiction/ page.
6. Use [[wiki-links]] for all cross-references within content.
7. Every page must follow the page anatomy below.

## Lint workflow
When asked to lint:
1. Review the index provided for [[wiki-links]] that point to slugs not in the index.
2. Identify any survey slugs not cross-referenced by at least one theme/ or entity/ page.
3. Return findings as a lint report. Do not call write_wiki_page during lint; report only.

## Page anatomy (all types)
Every page must include:
- A H1 title
- A frontmatter block: > Last updated: {date} · Sources: {survey slugs}
- ## Summary section
- ## Key statistics section (where applicable)
- ## Representative quotes section (open-ended responses only)
- ## Related pages section with [[wiki-links]]
```

### How the Wiki Maintainer LLM is Called

After Stage 5 (clustering) completes, the Python worker:

1. **Materialises a source document** — converts cluster summaries, per-question statistics, and representative quotes into a clean markdown string for the new survey
2. **Builds the context window:**
   - `SCHEMA.md` content (always included)
   - The wiki index: a list of existing slugs + one-line descriptions (not full page content — keeps tokens bounded regardless of wiki size)
   - The content of any existing pages that the new survey is likely to update (retrieved by semantic search against the source document, top 5–10 pages)
   - The new source document
3. **Calls Mistral Small** with tool calling enabled:
   - Tool: `write_wiki_page(slug, page_type, title, content)` — called once per page to create or update
4. **Executes tool calls:** for each `write_wiki_page` call, upsert to `wiki_pages`, recompute the page's embedding, extract `[[wiki-links]]` into `linked_slugs`, append to `wiki_log`

This is the full wiki update. No custom cross-reference code, no custom contradiction detection — that is handled entirely by Mistral following `SCHEMA.md`.

**Context window management:** Sending the full content of every existing wiki page is impractical once the wiki grows. The solution is the two-stage context: the index (bounded, ~1 line per page) tells Mistral what exists; it then identifies which pages to update, and only those pages' full content is loaded. This keeps token usage proportional to the number of relevant pages updated per ingest, not to wiki size.

**Embedding updates:** After each `write_wiki_page` upsert, if the content changed, recompute and store the embedding for that page. This keeps the wiki's HNSW index current.

### Re-ingesting an Updated Survey

If a survey is re-uploaded (corrected data), the ingestion pipeline reruns all stages from parse. At Stage 6, the wiki update call receives the new source document and the existing `survey/{slug}` page is updated with its full history. Theme and entity pages updated from this survey also get refreshed. The `survey_ids` array on each wiki page is what ties this together — when a survey is reprocessed, all wiki pages with that survey in `survey_ids` are candidates for update.

### The Compounding Loop

Good answers generated at query time can be filed back as `synthesis/` pages using the `file_to_wiki` tool in the query agent. This runs the same `write_wiki_page` upsert against `wiki_pages`. The wiki compounds not just through ingestion but through use.

The lint operation runs nightly: the same Mistral call but with `"Run the lint workflow on the wiki index below."` as the instruction.

---

## 7. Query Architecture

### Wiki as Primary Knowledge Source

At query time, the wiki is always consulted first. It contains pre-compiled narrative, cross-survey analysis, trend pages, and contradiction flags — everything the query agent needs for most questions. SQL and semantic search play supporting roles:

| Source | Role | When needed |
|---|---|---|
| `wiki_pages` | Pre-compiled narrative, themes, contradictions, trends | Every query |
| SQL (`survey_responses`) | Precise numbers, filters, breakdowns | Quantitative precision, demographic cuts |
| `open_ended_answers` (vector) | Verbatim quotes not in wiki | Quote requests, novel questions |
| `response_clusters` | Cluster summaries | Fallback if wiki update hasn't completed |

The Planner dispatches tools in parallel where they are needed; the Synthesiser merges the results. For most questions, wiki lookup alone is sufficient. For complex questions, SQL adds precision and semantic search adds verbatim texture.

```
User question
     │
     ▼
[Router — Mistral Small, free tier, <1 second]
  Classifies: Quantitative / Qualitative / Mixed / Trend / Meta
     │
     ▼
[Planner — Mistral Large]
  Determines which tools are needed; dispatches in parallel
     │
     ├─────────────────────┬──────────────────┬───────────────────────┐
     ▼                     ▼                  ▼                       ▼
[wiki_lookup]        [sql_query]       [semantic_search]      [cluster_summary]
 Primary: themes,     Precision:        Supporting:             Fallback only:
 trends, entities,    counts, %s,       verbatim quotes,        if wiki not yet
 contradictions,      breakdowns        novel queries           updated for this
 cross-references                                               survey
     │                     │                  │                       │
     └─────────────────────┴──────────────────┴───────────────────────┘
                                │
                                ▼
              [Synthesiser — Claude 3.5 Sonnet / Mistral Large]
               Merges streams into coherent narrative
               Cites sources: wiki page slugs + SQL results + quotes
                                │
                                ▼
                           User response
                                │
                     (if answer is substantive)
                                │
                                ▼
                      [file_to_wiki → synthesis/ page]
```

### Full Tool Inventory

| Tool | Description | Backed by |
|---|---|---|
| `wiki_lookup` | Semantic search over wiki pages + exact slug lookup | `wiki_pages` + HNSW index |
| `sql_query` | Validated SQL against structured survey data | PostgreSQL |
| `semantic_search` | Top-K open-ended answers by vector similarity | `open_ended_answers` + HNSW |
| `cluster_summary` | Pre-computed theme clusters for a question | `response_clusters` |
| `trend_analysis` | Cross-survey metric comparison by date range | SQL aggregation |
| `cross_survey_themes` | Clusters with high cosine similarity across surveys | `response_clusters.centroid` |
| `contradiction_check` | Retrieve known contradictions for a topic | `wiki_pages WHERE page_type='contradiction'` |
| `file_to_wiki` | File synthesiser output back as a synthesis page | `wiki_pages` upsert |

### Standard vs Deep Research Mode

| Mode | Behaviour |
|---|---|
| **Standard** | Wiki lookup + SQL (~3–5 seconds). Sufficient for most factual questions. |
| **Deep Research** | All tools in parallel + map-reduce over raw open-ended answers for novel queries. (~15–30 seconds) |

### Map-Reduce for Novel Queries (Deep Research)

When a question requires insight the wiki hasn't pre-compiled (e.g. a newly uploaded survey whose wiki update hasn't completed, or an unusual cross-cutting question):

1. **Map:** Split open-ended answers into batches of ~100; call Mistral Small to extract relevant themes from each batch in parallel
2. **Reduce:** Call Mistral Large to synthesise the batch summaries
3. **Cache** the result in `insight_cache` with a 24-hour TTL
4. **File to wiki** if the answer is substantive — so this expensive operation runs at most once per question

---

## 8. LLM Role Summary

| LLM | Role | Cost model |
|---|---|---|
| **Mistral Small** (free tier) | Router, question classifier, cluster labelling, LLMWiki ingest/lint operations, map-reduce batch pass | Free tier; all background/async so rate limits are acceptable |
| **Mistral Large** (`mistral-large-latest`) | Planner, map-reduce reduce step, complex reasoning | ~$0.003/1k tokens; used for reasoning-heavy steps |
| **Claude 3.5 Sonnet** | Final synthesis — the narrative answer the user reads | ~$0.015/1k tokens; called once per user query |

Using Mistral Small's free tier for all background work — including running the LLMWiki ingest and lint operations — keeps ongoing ingestion cost near zero. Mistral Large handles planning and reasoning where quality matters. Claude is reserved for the single final synthesis the user sees.

**Risk:** If Mistral Small's instruction-following proves inconsistent for complex wiki ingest operations (writing tool calls correctly, updating cross-references reliably), the wiki update stage can fall back to Claude Haiku (cheap, excellent instruction-following) while keeping Mistral for all other background tasks. The architecture is LLM-agnostic for this stage.

---

## 9. API Design

```
POST /api/ingest/upload            ← receive file, create ingestion_jobs row, return jobId
GET  /api/ingest/status/:jobId     ← poll stage + status for UI progress indicator
GET  /api/datasets                 ← list surveys with metadata
DELETE /api/datasets/:id           ← delete survey + cascade (responses, embeddings, wiki pages)

POST /api/chat                     ← user message → streaming response (SSE)
GET  /api/chat/sessions            ← list past sessions
GET  /api/chat/sessions/:id        ← retrieve full message history

GET  /api/wiki                     ← browse wiki index (slugs + one-line descriptions)
GET  /api/wiki/:slug               ← retrieve a single wiki page
```

All endpoints authenticated via Supabase Auth JWT. Row-level security ensures users access only their own surveys, jobs, and wiki pages.

The chat endpoint returns a **Server-Sent Events stream** — each tool result is flushed as it arrives, so the UI can show "Searching wiki…", "Running SQL…", "Retrieving quotes…" in real time before the synthesiser produces the final answer.

---

## 10. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Next.js 16, React 19, TypeScript | Already built |
| Database | Supabase (PostgreSQL 15+) | Structured data, wiki, job queue, RLS |
| Vector store | pgvector (HNSW index) | Co-located — responses, clusters, and wiki pages all indexed |
| File storage | Supabase Storage | Raw CSVs only (not wiki pages) |
| Wiki schema | `SCHEMA.md` in codebase | Version-controlled; read by worker at runtime |
| Embedding model | nomic-embed-text | Via Together AI API (open-source, 768-dim) |
| Clustering | HDBSCAN (Python scikit-learn-extra) | Background Python worker |
| Job queue | `ingestion_jobs` table + Python polling worker | No Redis dependency; survives restarts |
| Agent orchestration | Python FastAPI + LangGraph | LangGraph manages parallel tool dispatch cleanly |
| LLM: Router + Wiki | Mistral Small (free tier) | All background, latency-insensitive work |
| LLM: Planner + Reason | Mistral Large 2 | Complex reasoning, planning, reduce step |
| LLM: Synthesiser | Claude 3.5 Sonnet | Final user-facing narrative only |
| Hosting (API + worker) | Railway / Render | Python service + background worker process |
| Hosting (Frontend) | Vercel | Free tier sufficient |

---

## 11. Build Order

Steps 1–5 deliver a working MVP with SQL-backed insights. Steps 6–9 add the full compounding knowledge system.

| Step | Deliverable | What it unlocks |
|---|---|---|
| 1 | Supabase schema migrations, pgvector enabled | Foundation |
| 2 | Ingestion API: upload → parse → classify → SQL storage | Users can upload surveys |
| 3 | Embedding pipeline: async job → nomic → pgvector | Vector search enabled |
| 4 | SQL agent: NL → SQL → results | All quantitative questions answered |
| 5 | HDBSCAN clustering + Mistral cluster summaries | Qualitative theme queries via `cluster_summary` |
| 6 | **Wiki Maintainer**: source doc → Mistral + SCHEMA.md → wiki_pages | Knowledge compounds; cross-survey themes appear; `cluster_summary` becomes fallback |
| 7 | **Parallel query agent**: wiki + SQL + semantic simultaneously | Best possible context at query time |
| 8 | Synthesiser integration: Claude final answer + SSE streaming | Polished user-facing responses |
| 9 | Answer filing: synthesiser output → synthesis/ pages | Compounding loop closes |
| 10 | Frontend integration: swap mock services → real API | Full end-to-end product |
| 11 | Insight cache, wiki lint job (nightly) | Production robustness |
| 12 | Phase 2: events + participants tables + wiki extensions | Operational data layer |

---

## 12. Open Questions

- **Embedding hosting:** Together AI (easiest API) vs. self-hosted nomic via Ollama on Railway (cheapest at volume)? Together AI is the right default for MVP; revisit at scale.

- **Mistral Small instruction-following:** The wiki ingest operation requires the LLM to make multiple sequential tool calls correctly (write survey page, update theme pages, flag contradictions). If early testing shows this is unreliable with Mistral Small, fall back to Claude Haiku for the wiki stage only. Test this early with a real survey as the first schema validation step.

- **Wiki UI:** Should there be a browsable wiki view in the frontend? The `wiki_pages` API endpoint makes this straightforward, and it could help users understand why the system answered a question the way it did — a useful trust signal. Not required for MVP.

- **Answer filing consent:** Should users explicitly approve filing an answer back to the wiki as a `synthesis/` page, or should it be automatic for high-confidence responses? Explicit approval is safer for MVP.

- **Multi-tenancy:** Isolated data per parkrun region/team, or a shared corpus? RLS makes either easy. Shared corpus produces richer cross-region wiki pages. Decide before schema is finalised.

- **SCHEMA.md iteration:** The schema file will need tuning as real surveys are ingested. It should be reviewed and version-bumped before each major ingest batch. Changes should be tested with a sample survey in a staging environment before deploying to production — a bad schema update silently degrades every subsequent ingest.

- **Mistral free tier rate limits:** If ingest volume grows, the wiki update stage may need to be rate-limited or moved to a paid tier. The job queue supports configurable throughput; add a `min_interval_seconds` setting to the worker before this becomes a problem.
