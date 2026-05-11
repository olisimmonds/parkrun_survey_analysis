-- Persistent ingestion job queue.
-- Survives server restarts and browser tab closures.
-- The Python worker polls this table and advances each job stage-by-stage.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id    UUID NOT NULL REFERENCES surveys ON DELETE CASCADE,
  stage        TEXT NOT NULL CHECK (stage IN ('parse', 'classify', 'store', 'embed', 'cluster', 'wiki_update', 'done')),
  status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed')),
  attempt      INT NOT NULL DEFAULT 0,
  last_error   TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION update_ingestion_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ingestion_jobs_updated_at ON ingestion_jobs;
CREATE TRIGGER ingestion_jobs_updated_at
  BEFORE UPDATE ON ingestion_jobs
  FOR EACH ROW EXECUTE FUNCTION update_ingestion_jobs_updated_at();

-- Insight cache: stores expensive query results with a TTL.
-- Prevents re-running map-reduce over the same question within 24 hours.
CREATE TABLE IF NOT EXISTS insight_cache (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key  TEXT UNIQUE NOT NULL,
  result     JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL
);
