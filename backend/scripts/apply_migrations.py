"""
Apply all SQL migrations to the Supabase PostgreSQL database.

Usage:
  cd backend
  python scripts/apply_migrations.py

Requires psycopg2-binary and SUPABASE_DB_URL in .env:
  SUPABASE_DB_URL=postgresql://postgres.{ref}:{password}@aws-0-eu-west-2.pooler.supabase.com:5432/postgres

Get the password from:
  Supabase Dashboard → Project Settings → Database → Connection string

Also requires SUPABASE_DB_URL which you can find on the Supabase dashboard at:
  Project Settings → Database → URI (Direct connection)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIGRATIONS_DIR = ROOT / "migrations"


def _load_env() -> str:
    """Load SUPABASE_DB_URL from .env file or environment."""
    import os

    env_file = ROOT.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("SUPABASE_DB_URL="):
                return line.split("=", 1)[1].strip()

    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        print("ERROR: SUPABASE_DB_URL is not set.")
        print("Add it to your .env file:")
        print("  SUPABASE_DB_URL=postgresql://postgres.{ref}:{password}@...")
        print("Get the password from: Supabase Dashboard → Project Settings → Database")
        sys.exit(1)
    return url


def apply_migrations(db_url: str) -> None:
    try:
        import psycopg2  # type: ignore
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print("No migration files found in", MIGRATIONS_DIR)
        sys.exit(1)

    print(f"Connecting to Supabase PostgreSQL…")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True

    with conn.cursor() as cur:
        # Create a migrations tracking table if it doesn't exist.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        for migration_file in migration_files:
            filename = migration_file.name

            cur.execute("SELECT 1 FROM _migrations WHERE filename = %s", (filename,))
            if cur.fetchone():
                print(f"  [skip] {filename} (already applied)")
                continue

            print(f"  [apply] {filename}…", end=" ")
            sql = migration_file.read_text(encoding="utf-8")

            try:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO _migrations (filename) VALUES (%s)", (filename,)
                )
                print("OK")
            except Exception as exc:
                print(f"FAILED\n  Error: {exc}")
                conn.close()
                sys.exit(1)

    conn.close()
    print("\nAll migrations applied successfully.")


if __name__ == "__main__":
    db_url = _load_env()
    apply_migrations(db_url)
