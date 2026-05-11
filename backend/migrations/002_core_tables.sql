-- Survey catalogue: one row per uploaded survey file.
CREATE TABLE IF NOT EXISTS surveys (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name         TEXT NOT NULL,
  type         TEXT NOT NULL CHECK (type IN ('volunteer', 'participant', 'run_director', 'wellbeing', 'other')),
  source       TEXT CHECK (source IN ('surveymonkey', 'google_forms', 'manual')),
  conducted_at TIMESTAMPTZ,
  uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  uploaded_by  UUID,               -- future: REFERENCES auth.users
  row_count    INT,
  column_count INT,
  file_name    TEXT,
  metadata     JSONB DEFAULT '{}'
);

-- Questions extracted from survey column headers.
CREATE TABLE IF NOT EXISTS survey_questions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id     UUID NOT NULL REFERENCES surveys ON DELETE CASCADE,
  column_key    TEXT NOT NULL,      -- original CSV column name (snake_case key)
  label         TEXT NOT NULL,      -- human-readable label
  question_type TEXT NOT NULL CHECK (question_type IN ('rating', 'multiple_choice', 'open_ended', 'demographic', 'datetime', 'metadata')),
  position      INT NOT NULL,
  options       JSONB,              -- distinct values for multiple_choice questions
  UNIQUE (survey_id, column_key)
);

-- One row per survey respondent.
CREATE TABLE IF NOT EXISTS survey_responses (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  survey_id      UUID NOT NULL REFERENCES surveys ON DELETE CASCADE,
  respondent_ref TEXT,              -- anonymised participant ID from the source file
  responded_at   TIMESTAMPTZ,
  structured     JSONB NOT NULL DEFAULT '{}' -- { "question_id": value } for all non-open-ended answers
);

-- Open-ended text answers with embeddings for semantic search.
-- One row per (response, question) pair where question_type = 'open_ended'.
CREATE TABLE IF NOT EXISTS open_ended_answers (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  response_id   UUID NOT NULL REFERENCES survey_responses ON DELETE CASCADE,
  question_id   UUID NOT NULL REFERENCES survey_questions ON DELETE CASCADE,
  answer_text   TEXT NOT NULL,
  embedding     VECTOR(768),        -- populated async in Stage 4; NULL until then
  theme_cluster INT                 -- populated by clustering job in Stage 5; NULL until then
);
