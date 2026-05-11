-- Pre-computed HDBSCAN theme clusters.
-- One row per cluster per (survey, question) pair.
CREATE TABLE IF NOT EXISTS response_clusters (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id             UUID NOT NULL REFERENCES surveys ON DELETE CASCADE,
  question_id           UUID NOT NULL REFERENCES survey_questions ON DELETE CASCADE,
  cluster_id            INT NOT NULL,          -- HDBSCAN label (>= 0); -1 means noise, not stored
  label                 TEXT,                  -- LLM-generated theme label
  summary               TEXT,                  -- 2-3 sentence LLM summary
  response_count        INT NOT NULL,
  centroid              VECTOR(768),           -- mean of all embeddings in this cluster
  representative_quotes JSONB DEFAULT '[]',    -- [{"text": "...", "response_id": "..."}] — 3 closest to centroid
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (survey_id, question_id, cluster_id)
);
