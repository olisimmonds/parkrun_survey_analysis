"""
Smoke-test the three pgvector RPC functions in Supabase.

Usage (from the backend/ directory):
  python scripts/test_rpc.py

Requires:
  - SUPABASE_URL and SUPABASE_SERVICE_KEY in root .env
  - At least one survey ingested (for match_open_ended_answers / match_response_clusters)
  - Wiki pages written (for match_wiki_pages)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Resolve root .env so config.py finds it when run from any directory.
ROOT = Path(__file__).parent.parent.parent


async def main() -> None:
    # Bootstrap settings from root .env.
    import os
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    from supabase._async.client import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    db = await create_client(url, key)

    # Build a 768-dim zero vector (placeholder — no real embedding needed for smoke test).
    zero_vec = [0.0] * 768

    print("=" * 60)
    print("RPC smoke tests")
    print("=" * 60)

    # ── 1. match_wiki_pages ────────────────────────────────────────────────
    print("\n[1] match_wiki_pages …", end=" ", flush=True)
    try:
        result = await db.rpc(
            "match_wiki_pages",
            {"query_embedding": zero_vec, "match_count": 3},
        ).execute()
        rows = result.data or []
        print(f"OK — {len(rows)} row(s) returned")
        for r in rows[:2]:
            sim = float(r.get("similarity") or 0)
            print(f"    slug={r.get('slug')!r}  similarity={sim:.4f}")
    except Exception as exc:
        print(f"FAILED: {exc}")

    # ── 2. match_open_ended_answers ───────────────────────────────────────
    print("\n[2] match_open_ended_answers …", end=" ", flush=True)
    try:
        result = await db.rpc(
            "match_open_ended_answers",
            {"query_embedding": zero_vec, "match_count": 3, "filter_survey_ids": []},
        ).execute()
        rows = result.data or []
        print(f"OK — {len(rows)} row(s) returned")
        for r in rows[:2]:
            snippet = (r.get("answer_text") or "")[:60]
            sim = float(r.get("similarity") or 0)
            print(f"    answer={snippet!r}  similarity={sim:.4f}")
    except Exception as exc:
        print(f"FAILED: {exc}")

    # ── 3. match_response_clusters ────────────────────────────────────────
    print("\n[3] match_response_clusters …", end=" ", flush=True)
    try:
        result = await db.rpc(
            "match_response_clusters",
            {"query_embedding": zero_vec, "match_count": 3},
        ).execute()
        rows = result.data or []
        print(f"OK — {len(rows)} row(s) returned")
        for r in rows[:2]:
            print(f"    label={r.get('label')!r}  responses={r.get('response_count')}")
    except Exception as exc:
        print(f"FAILED: {exc}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
