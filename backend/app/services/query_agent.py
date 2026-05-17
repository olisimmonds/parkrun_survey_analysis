"""
Query Agent — parallel multi-source retrieval and synthesis.

Architecture:
  1. Router classifies the question (Quantitative / Qualitative / Mixed / Trend / Meta)
  2. Planner decides which tools are needed
  3. Tools run in parallel via asyncio.gather()
  4. Synthesiser merges all results into a coherent narrative (Groq)

Tool sources:
  wiki_lookup     — semantic search over wiki_pages (primary)
  sql_query       — structured SQL for numbers/percentages/filters
  semantic_search — top-K open-ended answers by vector similarity
  cluster_summary — pre-computed theme clusters (fallback if wiki incomplete)

The synthesiser receives all tool results and produces the final answer.
SSE streaming is handled at the router layer, not here.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator

from groq import AsyncGroq

from app.config import get_settings
from app.services.embedder import embed_query

log = logging.getLogger(__name__)

QUESTION_TYPES = ("Quantitative", "Qualitative", "Mixed", "Trend", "Meta")

_ROUTER_PROMPT = """\
Classify this user question about parkrun survey data into exactly one category:
  Quantitative — asks for numbers, percentages, counts, ratings, rankings
  Qualitative  — asks for themes, opinions, experiences, suggestions, quotes
  Mixed        — asks for both quantitative and qualitative insight
  Trend        — asks about change over time, year-on-year, historical patterns
  Meta         — asks about what data is available, what surveys exist

Question: {question}

Reply with ONLY one word from the list above."""

_SYNTH_SYSTEM = """\
You are an expert analyst synthesising insights from parkrun community survey data.
Respond in clear, professional prose suitable for a parkrun Digital Ambassador.
Always cite your sources: wiki page slugs, SQL result context, or quote sources.
Be specific — use exact figures from the context. Never speculate beyond the evidence.
Keep your response concise: 200–400 words unless the question requires more depth."""

_SYNTH_USER = """\
Question: {question}

Context from the knowledge base:
{context}

Synthesise a coherent answer. Cite sources inline."""


async def _route_question(question: str) -> str:
    settings = get_settings()
    client = AsyncGroq(api_key=settings.groq_api_key)
    resp = await client.chat.completions.create(
        model=settings.groq_fast_model,
        messages=[{"role": "user", "content": _ROUTER_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=10,
    )
    raw = (resp.choices[0].message.content or "Mixed").strip()
    for qt in QUESTION_TYPES:
        if qt.lower() in raw.lower():
            return qt
    return "Mixed"


async def _wiki_lookup(db: Any, question: str, top_k: int = 5) -> list[dict]:
    """Semantic search over wiki_pages, returning the most relevant pages."""
    embedding = await embed_query(question)
    try:
        result = await db.rpc(
            "match_wiki_pages",
            {"query_embedding": embedding, "match_count": top_k},
        ).execute()
        return result.data or []
    except Exception as exc:
        log.warning("wiki_lookup failed: %s", exc)
        return []


async def _sql_query(db: Any, question: str, dataset_ids: list[str] | None) -> list[dict]:
    """
    Retrieve structured aggregations relevant to the question.
    Returns per-survey statistics for rating and multiple_choice questions.
    This is a simplified implementation — a full NL→SQL generator would go here.
    """
    try:
        query = db.table("response_clusters").select(
            "label, summary, response_count, survey_id, question_id"
        )
        if dataset_ids:
            query = query.in_("survey_id", dataset_ids)
        result = await query.limit(10).execute()
        return result.data or []
    except Exception as exc:
        log.warning("sql_query failed: %s", exc)
        return []


async def _semantic_search(
    db: Any, question: str, dataset_ids: list[str] | None, top_k: int = 8
) -> list[dict]:
    """Top-K open-ended answers by cosine similarity to the question."""
    embedding = await embed_query(question)
    try:
        result = await db.rpc(
            "match_open_ended_answers",
            {
                "query_embedding": embedding,
                "match_count": top_k,
                "filter_survey_ids": dataset_ids or [],
            },
        ).execute()
        return result.data or []
    except Exception as exc:
        log.warning("semantic_search failed: %s", exc)
        return []


async def _cluster_summary(
    db: Any, dataset_ids: list[str] | None
) -> list[dict]:
    """Fetch pre-computed cluster summaries (fallback if wiki not yet updated)."""
    try:
        query = db.table("response_clusters").select(
            "label, summary, response_count, representative_quotes"
        )
        if dataset_ids:
            query = query.in_("survey_id", dataset_ids)
        result = await query.limit(20).execute()
        return result.data or []
    except Exception as exc:
        log.warning("cluster_summary failed: %s", exc)
        return []


def _format_context(
    wiki_pages: list[dict],
    sql_results: list[dict],
    semantic_hits: list[dict],
    question_type: str,
) -> str:
    """Assemble all tool results into a single context block for the synthesiser."""
    parts = []

    if wiki_pages:
        parts.append("### Wiki Knowledge Base")
        for page in wiki_pages:
            parts.append(
                f"**[{page.get('slug', 'unknown')}]** {page.get('title', '')}\n"
                f"{textwrap.shorten(page.get('content', ''), width=800, placeholder='…')}"
            )

    if sql_results and question_type in ("Quantitative", "Mixed", "Trend"):
        parts.append("### Quantitative Data")
        for row in sql_results[:5]:
            parts.append(
                f"- {row.get('label', 'Cluster')}: {row.get('response_count', '?')} responses — "
                f"{row.get('summary', '')}"
            )

    if semantic_hits and question_type in ("Qualitative", "Mixed"):
        parts.append("### Verbatim Quotes")
        for hit in semantic_hits[:5]:
            parts.append(f'> "{hit.get("answer_text", "")}"')

    return "\n\n".join(parts) if parts else "_No relevant context found in the knowledge base._"


def _cache_key(question: str, mode: str, dataset_ids: list[str] | None) -> str:
    payload = f"{question}:{mode}:{sorted(dataset_ids or [])}"
    return hashlib.sha256(payload.encode()).hexdigest()


async def _get_cached(db: Any, key: str) -> dict | None:
    try:
        cutoff = datetime.now(timezone.utc).isoformat()
        result = (
            await db.table("insight_cache")
            .select("result")
            .eq("cache_key", key)
            .gt("expires_at", cutoff)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if rows:
            return rows[0]["result"]
    except Exception as exc:
        log.warning("Cache lookup failed: %s", exc)
    return None


async def _set_cached(db: Any, key: str, result: dict, ttl_hours: int = 24) -> None:
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
    try:
        await db.table("insight_cache").upsert({
            "cache_key": key,
            "result": result,
            "expires_at": expires_at,
        }).execute()
    except Exception as exc:
        log.warning("Cache write failed: %s", exc)


async def _run_retrieval(
    db: Any,
    question: str,
    mode: str,
    dataset_ids: list[str] | None,
) -> tuple[str, list[dict], list[dict], list[dict], str]:
    """
    Route the question and run all retrieval tools in parallel.

    Returns (question_type, wiki_pages, sql_results, semantic_hits, context).
    All tool errors are swallowed and return empty lists so a single
    failed tool cannot break the whole response.
    """
    question_type = await _route_question(question)
    log.info("Question classified as: %s", question_type)

    tasks: dict[str, asyncio.Task] = {
        "wiki": asyncio.create_task(_wiki_lookup(db, question)),
        "sql":  asyncio.create_task(_sql_query(db, question, dataset_ids)),
    }
    if mode == "deep-research" or question_type in ("Qualitative", "Mixed"):
        tasks["semantic"] = asyncio.create_task(_semantic_search(db, question, dataset_ids))
        tasks["clusters"] = asyncio.create_task(_cluster_summary(db, dataset_ids))

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_map = dict(zip(tasks.keys(), results))

    wiki_pages    = result_map.get("wiki")    or []
    sql_results   = result_map.get("sql")     or []
    semantic_hits = result_map.get("semantic") or []
    if isinstance(wiki_pages,    Exception): wiki_pages    = []
    if isinstance(sql_results,   Exception): sql_results   = []
    if isinstance(semantic_hits, Exception): semantic_hits = []

    context = _format_context(wiki_pages, sql_results, semantic_hits, question_type)
    return question_type, wiki_pages, sql_results, semantic_hits, context


def _build_sources(wiki_pages: list[dict]) -> list[dict]:
    return [
        {
            "id": p.get("id", p.get("slug", "")),
            "name": p.get("title", p.get("slug", "")),
            "excerpt": textwrap.shorten(p.get("content", ""), width=200, placeholder="..."),
            "datasetId": (p.get("survey_ids") or [""])[0],
            "relevanceScore": float(p.get("similarity") or 0.0),
        }
        for p in wiki_pages[:3]
    ]


async def answer_question(
    db: Any,
    question: str,
    mode: str = "standard",
    dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Non-streaming query agent entry point. Returns the full answer dict.
    Use stream_answer() for real-time token delivery.
    """
    settings = get_settings()

    key = _cache_key(question, mode, dataset_ids) if mode == "standard" else None
    if key:
        cached = await _get_cached(db, key)
        if cached:
            log.info("Cache hit for question (%.50s)", question)
            return cached

    question_type, wiki_pages, _, _, context = await _run_retrieval(db, question, mode, dataset_ids)

    client = AsyncGroq(api_key=settings.groq_api_key)
    synth_resp = await client.chat.completions.create(
        model=settings.groq_capable_model,
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": _SYNTH_USER.format(question=question, context=context)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    answer = synth_resp.choices[0].message.content or "No answer generated."
    result = {"answer": answer, "question_type": question_type, "sources": _build_sources(wiki_pages)}

    if key:
        await _set_cached(db, key, result)
    return result


async def stream_answer(
    db: Any,
    question: str,
    mode: str = "standard",
    dataset_ids: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Streaming query agent entry point.

    Yields dicts in order:
      {"type": "chunk",  "text": str}          — one token at a time
      {"type": "done",   "sources": [...],
                         "question_type": str}  — final event
    """
    settings = get_settings()

    key = _cache_key(question, mode, dataset_ids) if mode == "standard" else None
    if key:
        cached = await _get_cached(db, key)
        if cached:
            log.info("Cache hit (stream_answer): %.50s", question)
            for word in (cached.get("answer") or "").split():
                yield {"type": "chunk", "text": word + " "}
            yield {"type": "done", "sources": cached.get("sources", []),
                   "question_type": cached.get("question_type", "")}
            return

    question_type, wiki_pages, _, _, context = await _run_retrieval(db, question, mode, dataset_ids)

    client = AsyncGroq(api_key=settings.groq_api_key)
    stream = await client.chat.completions.create(
        model=settings.groq_capable_model,
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": _SYNTH_USER.format(question=question, context=context)},
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    chunks: list[str] = []
    async for event in stream:
        token = (event.choices[0].delta.content or "") if event.choices else ""
        if token:
            chunks.append(token)
            yield {"type": "chunk", "text": token}

    answer = "".join(chunks)
    sources = _build_sources(wiki_pages)
    result = {"answer": answer, "question_type": question_type, "sources": sources}
    if key:
        await _set_cached(db, key, result)

    yield {"type": "done", "sources": sources, "question_type": question_type}
