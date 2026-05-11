"""
Supabase async client — one instance per process, shared via FastAPI dependency.

The service-role key bypasses Row-Level Security and is used by the worker and
internal API paths. The anon key is used for user-facing operations once auth
is wired up in Phase 2.
"""
from functools import lru_cache

from supabase._async.client import AsyncClient, create_client  # type: ignore[attr-defined]

from app.config import get_settings


@lru_cache
def _get_service_client() -> AsyncClient:
    """Cached async Supabase client using the service-role key."""
    settings = get_settings()
    # create_client is synchronous; the returned AsyncClient uses async methods.
    import asyncio

    async def _build() -> AsyncClient:
        return await create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )

    return asyncio.get_event_loop().run_until_complete(_build())


async def get_db() -> AsyncClient:
    """FastAPI dependency that yields the shared Supabase service client."""
    return _get_service_client()
