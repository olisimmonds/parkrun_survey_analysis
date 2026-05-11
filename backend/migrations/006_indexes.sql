-- HNSW vector indexes for fast approximate nearest-neighbour search.
-- m=16, ef_construction=64 is a good default for datasets up to ~10M vectors.
-- Increase ef_construction to 128 for better recall at the cost of build time.

CREATE INDEX IF NOT EXISTS open_ended_answers_embedding_idx
  ON open_ended_answers
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS response_clusters_centroid_idx
  ON response_clusters
  USING hnsw (centroid vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS wiki_pages_embedding_idx
  ON wiki_pages
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Structured query indexes.
CREATE INDEX IF NOT EXISTS survey_responses_survey_id_idx ON survey_responses (survey_id);
CREATE INDEX IF NOT EXISTS open_ended_answers_question_cluster_idx ON open_ended_answers (question_id, theme_cluster);
CREATE INDEX IF NOT EXISTS surveys_type_conducted_idx ON surveys (type, conducted_at);
CREATE INDEX IF NOT EXISTS wiki_pages_page_type_idx ON wiki_pages (page_type);
CREATE INDEX IF NOT EXISTS wiki_pages_slug_idx ON wiki_pages (slug);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_stage_idx ON ingestion_jobs (status, stage);
CREATE INDEX IF NOT EXISTS insight_cache_expires_idx ON insight_cache (expires_at);
