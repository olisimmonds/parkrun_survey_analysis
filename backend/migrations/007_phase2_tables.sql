-- Phase 2: event attendance and participant history.
-- Not required for MVP. Tables are created here so the schema is stable
-- for Phase 2 integration without any migration required.

CREATE TABLE IF NOT EXISTS events (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_name   TEXT NOT NULL,
  location     TEXT,
  event_date   DATE NOT NULL,
  finishers    INT,
  volunteers   INT,
  first_timers INT,
  metadata     JSONB DEFAULT '{}',
  UNIQUE (event_name, event_date)
);

CREATE TABLE IF NOT EXISTS participants (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  athlete_ref    TEXT UNIQUE NOT NULL,  -- anonymised parkrun athlete ID
  home_event     TEXT,
  total_runs     INT DEFAULT 0,
  first_run_date DATE,
  metadata       JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS events_date_idx ON events (event_date);
CREATE INDEX IF NOT EXISTS events_name_date_idx ON events (event_name, event_date);
