-- pgvector similarity search functions used by the query agent.
-- These expose RPC endpoints callable via supabase-py's db.rpc().

-- Match wiki pages by semantic similarity to a query embedding.
CREATE OR REPLACE FUNCTION match_wiki_pages(
  query_embedding VECTOR(768),
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  id          UUID,
  slug        TEXT,
  title       TEXT,
  page_type   TEXT,
  content     TEXT,
  survey_ids  UUID[],
  updated_at  TIMESTAMPTZ,
  similarity  FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    wp.id,
    wp.slug,
    wp.title,
    wp.page_type,
    wp.content,
    wp.survey_ids,
    wp.updated_at,
    1 - (wp.embedding <=> query_embedding) AS similarity
  FROM wiki_pages wp
  WHERE wp.embedding IS NOT NULL
  ORDER BY wp.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Match open-ended answers by semantic similarity.
-- Optionally filter to specific survey IDs.
CREATE OR REPLACE FUNCTION match_open_ended_answers(
  query_embedding   VECTOR(768),
  match_count       INT DEFAULT 8,
  filter_survey_ids UUID[] DEFAULT '{}'
)
RETURNS TABLE (
  id          UUID,
  answer_text TEXT,
  question_id UUID,
  response_id UUID,
  similarity  FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    oea.id,
    oea.answer_text,
    oea.question_id,
    oea.response_id,
    1 - (oea.embedding <=> query_embedding) AS similarity
  FROM open_ended_answers oea
  WHERE
    oea.embedding IS NOT NULL
    AND (
      array_length(filter_survey_ids, 1) IS NULL
      OR oea.response_id IN (
        SELECT sr.id FROM survey_responses sr
        WHERE sr.survey_id = ANY(filter_survey_ids)
      )
    )
  ORDER BY oea.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Match response clusters by centroid similarity (cross-survey theme search).
CREATE OR REPLACE FUNCTION match_response_clusters(
  query_embedding VECTOR(768),
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  id             UUID,
  survey_id      UUID,
  question_id    UUID,
  label          TEXT,
  summary        TEXT,
  response_count INT,
  similarity     FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    rc.id,
    rc.survey_id,
    rc.question_id,
    rc.label,
    rc.summary,
    rc.response_count,
    1 - (rc.centroid <=> query_embedding) AS similarity
  FROM response_clusters rc
  WHERE rc.centroid IS NOT NULL
  ORDER BY rc.centroid <=> query_embedding
  LIMIT match_count;
END;
$$;
