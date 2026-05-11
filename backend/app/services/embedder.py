"""
Embedding service.

Supports two providers:
  local    — sentence-transformers running in-process (no API key required).
             Uses nomic-ai/nomic-embed-text-v1.5 (768 dims, MTEB-competitive).
  together — Together AI hosted nomic-embed-text via REST API (requires
             TOGETHER_API_KEY; identical model, useful for production).

The provider is selected via the EMBEDDING_PROVIDER env var (default: "local").

nomic-embed-text-v1.5 requires a task-type prefix on input text:
  Documents:  "search_document: {text}"
  Queries:    "search_query: {text}"
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
import numpy as np

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 256


@lru_cache
def _get_local_model() -> "SentenceTransformer":
    """Load sentence-transformers model once and cache it."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    settings = get_settings()
    log.info("Loading local embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, trust_remote_code=True)
    return model


def _embed_local(texts: list[str], prefix: str) -> list[list[float]]:
    model = _get_local_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()  # type: ignore[return-value]


async def _embed_together(texts: list[str], prefix: str) -> list[list[float]]:
    settings = get_settings()
    if not settings.together_api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set but embedding_provider='together'")

    prefixed = [f"{prefix}{t}" for t in texts]
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.together.xyz/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.together_api_key}"},
            json={"model": settings.embedding_model, "input": prefixed},
        )
        resp.raise_for_status()
        data = resp.json()

    # Together AI returns: {"data": [{"embedding": [...], "index": 0}, ...]}
    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts (survey responses, wiki page content).
    Returns a list of 768-dimensional float vectors.
    Processes in batches to respect memory and API limits.
    """
    settings = get_settings()
    results: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        if settings.embedding_provider == "together":
            batch_result = await _embed_together(batch, DOCUMENT_PREFIX)
        else:
            # Run CPU-bound local model in a thread pool to avoid blocking the event loop.
            batch_result = await asyncio.get_event_loop().run_in_executor(
                None, _embed_local, batch, DOCUMENT_PREFIX
            )
        results.extend(batch_result)

    return results


async def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at query time for semantic search)."""
    settings = get_settings()
    if settings.embedding_provider == "together":
        results = await _embed_together([text], QUERY_PREFIX)
    else:
        results = await asyncio.get_event_loop().run_in_executor(
            None, _embed_local, [text], QUERY_PREFIX
        )
    return results[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
