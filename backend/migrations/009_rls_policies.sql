-- Row Level Security policies for multi-user production deployment.
--
-- Strategy: all tables are readable/writable only by the service role
-- (used by the FastAPI backend) or authenticated users.
-- The anon role (used by direct browser connections) has NO access.
--
-- Apply AFTER your application is live and you have confirmed the service
-- role key is being used exclusively by the backend.

-- Enable RLS on all data tables.
ALTER TABLE surveys           ENABLE ROW LEVEL SECURITY;
ALTER TABLE survey_questions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE survey_responses  ENABLE ROW LEVEL SECURITY;
ALTER TABLE open_ended_answers ENABLE ROW LEVEL SECURITY;
ALTER TABLE response_clusters  ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki_pages         ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki_log           ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_jobs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE insight_cache      ENABLE ROW LEVEL SECURITY;

-- Service role bypass: Supabase service role always bypasses RLS by default,
-- but these explicit policies make intent clear and survive RLS resets.

CREATE POLICY "service role full access" ON surveys
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON survey_questions
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON survey_responses
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON open_ended_answers
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON response_clusters
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON wiki_pages
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON wiki_log
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON ingestion_jobs
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service role full access" ON insight_cache
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Authenticated users (Phase 2: scoped to their own org)
-- For MVP, authenticated users get full read access and no write access
-- (all writes go through the backend service layer).

CREATE POLICY "authenticated read surveys" ON surveys
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "authenticated read wiki_pages" ON wiki_pages
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "authenticated read response_clusters" ON response_clusters
  FOR SELECT TO authenticated USING (true);

-- Anon role: no access (deny by default — Supabase denies if no policy matches).
