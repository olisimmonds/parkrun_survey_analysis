-- Wiki pages: the LLMWiki compounding knowledge base.
-- One row per page. Pages are created and updated by the wiki maintainer LLM.
CREATE TABLE IF NOT EXISTS wiki_pages (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug         TEXT UNIQUE NOT NULL,   -- e.g. 'theme/volunteer-motivation'
  page_type    TEXT NOT NULL CHECK (page_type IN ('survey', 'theme', 'entity', 'trend', 'contradiction', 'synthesis')),
  title        TEXT NOT NULL,
  content      TEXT NOT NULL,          -- full markdown body
  embedding    VECTOR(768),            -- embedding of content for semantic wiki search
  linked_slugs TEXT[] DEFAULT '{}',   -- outbound [[wiki-links]] extracted from content
  survey_ids   UUID[] DEFAULT '{}',   -- surveys this page draws evidence from
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Trigger to auto-update updated_at on every modification.
CREATE OR REPLACE FUNCTION update_wiki_pages_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS wiki_pages_updated_at ON wiki_pages;
CREATE TRIGGER wiki_pages_updated_at
  BEFORE UPDATE ON wiki_pages
  FOR EACH ROW EXECUTE FUNCTION update_wiki_pages_updated_at();

-- Append-only operation log. Records every ingest, lint, and answer-filing event.
CREATE TABLE IF NOT EXISTS wiki_log (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type TEXT NOT NULL CHECK (event_type IN ('ingest', 'update', 'lint', 'filed_answer')),
  page_slug  TEXT,
  survey_id  UUID REFERENCES surveys ON DELETE SET NULL,
  summary    TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
