"""
Supabase async client — one shared instance per process.

Initialised lazily on first request. The lru_cache + run_until_complete
approach won't work inside an already-running event loop, so we use a
simple module-level singleton created on first await.
"""
from __future__ import annotations

import asyncio
import logging

from supabase._async.client import AsyncClient, create_client  # type: ignore[attr-defined]

from app.config import get_settings

log = logging.getLogger(__name__)

_client: AsyncClient | None = None
_lock: asyncio.Lock | None = None


async def get_db() -> AsyncClient:
    """FastAPI dependency that returns the shared Supabase service client."""
    global _client, _lock

    if _client is not None:
        return _client

    # Create the lock inside the running loop (safe on any Python ≥ 3.10).
    if _lock is None:
        _lock = asyncio.Lock()

    async with _lock:
        if _client is None:
            settings = get_settings()
            log.info("Creating Supabase async client…")
            _client = await create_client(
                settings.supabase_url,
                settings.supabase_service_key,
            )
            log.info("Supabase client ready.")

    return _client
