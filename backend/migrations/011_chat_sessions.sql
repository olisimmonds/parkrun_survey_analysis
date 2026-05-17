-- Persistent chat sessions.
-- Replaces the in-memory _sessions dict in chat.py.
-- Messages are stored as a JSONB array so no join is needed on read.
CREATE TABLE IF NOT EXISTS chat_sessions (
  id         UUID PRIMARY KEY,
  title      TEXT NOT NULL,
  mode       TEXT NOT NULL DEFAULT 'standard',
  messages   JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION _update_chat_sessions_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS chat_sessions_updated_at ON chat_sessions;
CREATE TRIGGER chat_sessions_updated_at
  BEFORE UPDATE ON chat_sessions
  FOR EACH ROW EXECUTE FUNCTION _update_chat_sessions_updated_at();

-- Service role has full access; authenticated users can only read their own sessions.
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service role full access" ON chat_sessions
  FOR ALL TO service_role USING (true) WITH CHECK (true);
