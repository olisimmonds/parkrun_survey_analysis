"""
Wiki Maintainer — LLMWiki ingest and lint operations.

Implements Karpathy's LLMWiki pattern for the parkrun knowledge base.
The wiki lives in the wiki_pages PostgreSQL table (not the filesystem),
but the compounding behaviour is identical.

Ingest flow
───────────
1. Materialise a source document from cluster summaries + SQL statistics.
2. Build a context window:
   - SCHEMA.md content (always included)
   - Wiki index: existing slugs + titles (bounded, never grows unboundedly)
   - Relevant existing pages (top 5 by semantic similarity to source doc)
   - New source document
3. Call Groq with tool calling enabled:
   - write_wiki_page(slug, page_type, title, content)
   - finish_ingest(summary)
4. Execute tool calls: upsert wiki_pages rows, recompute embeddings, extract links.

Lint flow (nightly)
───────────────────
Same context but a different instruction — no write_wiki_page calls, just a report.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from uuid import UUID

from groq import AsyncGroq

from app.config import get_settings
from app.services.embedder import embed_documents, embed_query

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "wiki" / "SCHEMA.md"

VALID_PAGE_TYPES = frozenset(
    {"survey", "theme", "entity", "trend", "contradiction", "synthesis"}
)

_WRITE_WIKI_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_wiki_page",
        "description": "Create or update a wiki page in the parkrun knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Page slug, e.g. 'theme/volunteer-motivation'",
                },
                "page_type": {
                    "type": "string",
                    "enum": list(VALID_PAGE_TYPES),
                },
                "title": {"type": "string"},
                "content": {
                    "type": "string",
                    "description": "Full markdown body following the SCHEMA.md page anatomy.",
                },
            },
            "required": ["slug", "page_type", "title", "content"],
        },
    },
}

_FINISH_INGEST_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_ingest",
        "description": "Signal that the ingest operation is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary: N pages created, M pages updated, contradictions flagged.",
                }
            },
            "required": ["summary"],
        },
    },
}


def _load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def _extract_wiki_links(content: str) -> list[str]:
    """Extract all [[slug]] references from markdown content."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


def _build_source_document(
    survey_name: str,
    survey_type: str,
    conducted_at: str | None,
    row_count: int,
    clusters: list[dict[str, Any]],
    sql_stats: dict[str, Any],
) -> str:
    """
    Materialise a structured markdown source document from ingestion outputs.
    This is what the wiki maintainer LLM reads as its primary input.
    """
    lines = [
        f"# Source Document: {survey_name}",
        f"\n**Survey type:** {survey_type}",
        f"**Period:** {conducted_at or 'Unknown'}",
        f"**Total responses:** {row_count}",
        "",
        "## Quantitative Summary",
    ]

    if sql_stats:
        for question, stats in sql_stats.items():
            lines.append(f"\n### {question}")
            if "mean" in stats:
                lines.append(f"- Mean rating: {stats['mean']:.2f}")
            if "distribution" in stats:
                for val, count in stats["distribution"].items():
                    pct = 100 * count / row_count if row_count else 0
                    lines.append(f"- {val}: {count} ({pct:.1f}%)")
    else:
        lines.append("_No quantitative statistics available._")

    lines.append("\n## Qualitative Theme Clusters")

    if clusters:
        for cluster in clusters:
            q_label = cluster.get("question_label", "Question")
            lines.append(f"\n### {q_label} — {cluster.get('label', 'Theme')}")
            lines.append(f"**Responses in cluster:** {cluster.get('response_count', 0)}")
            if cluster.get("summary"):
                lines.append(f"\n{cluster['summary']}")
            quotes = cluster.get("representative_quotes", [])
            if quotes:
                lines.append("\n**Representative quotes:**")
                for q in quotes[:3]:
                    text = q.get("text", q) if isinstance(q, dict) else q
                    lines.append(f'> "{text}"')
    else:
        lines.append("_No qualitative clusters available._")

    return "\n".join(lines)


async def run_ingest(
    db: Any,
    survey_id: UUID,
    survey_name: str,
    survey_type: str,
    conducted_at: str | None,
    row_count: int,
    clusters: list[dict[str, Any]],
    sql_stats: dict[str, Any],
) -> str:
    """
    Run a full wiki ingest for a newly processed survey.
    Returns a summary string of what was created/updated.
    """
    settings = get_settings()
    schema = _load_schema()

    source_doc = _build_source_document(
        survey_name, survey_type, conducted_at, row_count, clusters, sql_stats
    )

    # Fetch the wiki index (slugs + titles only — bounded context).
    index_result = (
        await db.table("wiki_pages")
        .select("slug, title, page_type, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    index_entries = index_result.data or []
    index_text = "\n".join(
        f"- [{e['page_type']}] {e['slug']}: {e['title']}"
        for e in index_entries
    )

    # Fetch the 5 most relevant existing pages by semantic similarity.
    query_embedding = await embed_query(source_doc[:2000])
    similar_result = (
        await db.rpc(
            "match_wiki_pages",
            {"query_embedding": query_embedding, "match_count": 5},
        ).execute()
    )
    relevant_pages = similar_result.data or []
    relevant_text = "\n\n".join(
        f"---\n## Existing page: {p['slug']}\n\n{p['content']}"
        for p in relevant_pages
    )

    system_msg = (
        f"{schema}\n\n"
        "---\n\n"
        "## Current wiki index\n\n"
        f"{index_text or '_Empty wiki — no pages yet._'}\n\n"
        "---\n\n"
        f"## Relevant existing pages\n\n{relevant_text or '_None retrieved._'}"
    )

    user_msg = (
        "Run the ingest workflow on the source document below. "
        "Use write_wiki_page to create or update all relevant pages, "
        "then call finish_ingest.\n\n"
        f"{source_doc}"
    )

    client = AsyncGroq(api_key=settings.groq_api_key)
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Agentic loop: keep calling the LLM until it calls finish_ingest.
    pages_written: list[str] = []
    finish_summary = ""
    max_rounds = 10

    for _round in range(max_rounds):
        response = await client.chat.completions.create(
            model=settings.groq_capable_model,
            messages=messages,
            tools=[_WRITE_WIKI_PAGE_TOOL, _FINISH_INGEST_TOOL],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=4096,
        )

        msg = response.choices[0].message
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            break

        tool_results = []
        done = False
        for tool_call in msg.tool_calls:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn == "write_wiki_page":
                result = await _execute_write_wiki_page(db, survey_id, args)
                pages_written.append(args["slug"])
                tool_results.append({"tool_call_id": tool_call.id, "content": result})

            elif fn == "finish_ingest":
                finish_summary = args.get("summary", "")
                tool_results.append({"tool_call_id": tool_call.id, "content": "Ingest complete."})
                done = True

        messages.append({"role": "tool", "content": json.dumps(tool_results)})

        if done:
            break

    # Log the ingest operation.
    await db.table("wiki_log").insert({
        "event_type": "ingest",
        "survey_id": str(survey_id),
        "summary": finish_summary or f"Wrote {len(pages_written)} pages: {', '.join(pages_written[:5])}",
    }).execute()

    log.info("Wiki ingest complete for survey %s: %d pages written.", survey_id, len(pages_written))
    return finish_summary


async def _execute_write_wiki_page(
    db: Any, survey_id: UUID, args: dict[str, Any]
) -> str:
    """Upsert a wiki page, recompute its embedding, and extract wiki-links."""
    slug = args["slug"]
    content = args["content"]

    if args.get("page_type") not in VALID_PAGE_TYPES:
        return f"Error: invalid page_type '{args.get('page_type')}'"

    linked_slugs = _extract_wiki_links(content)
    embeddings = await embed_documents([content])
    embedding = embeddings[0]

    row = {
        "slug": slug,
        "page_type": args["page_type"],
        "title": args["title"],
        "content": content,
        "embedding": embedding,
        "linked_slugs": linked_slugs,
        "survey_ids": [str(survey_id)],
    }

    existing = (
        await db.table("wiki_pages").select("id, survey_ids").eq("slug", slug).execute()
    )
    if existing.data:
        existing_ids = existing.data[0].get("survey_ids") or []
        if str(survey_id) not in existing_ids:
            existing_ids.append(str(survey_id))
        row["survey_ids"] = existing_ids
        await db.table("wiki_pages").update(row).eq("slug", slug).execute()
        log.debug("Updated wiki page: %s", slug)
        return f"Updated page: {slug}"
    else:
        await db.table("wiki_pages").insert(row).execute()
        log.debug("Created wiki page: %s", slug)
        return f"Created page: {slug}"


async def run_lint(db: Any) -> str:
    """
    Run a wiki lint check. Returns a lint report as a markdown string.
    Does NOT modify any wiki pages.
    """
    settings = get_settings()
    schema = _load_schema()

    index_result = await db.table("wiki_pages").select("slug, title, page_type").execute()
    index_entries = index_result.data or []
    all_slugs = {e["slug"] for e in index_entries}
    index_text = "\n".join(
        f"- [{e['page_type']}] {e['slug']}: {e['title']}" for e in index_entries
    )

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.groq_capable_model,
        messages=[
            {"role": "system", "content": schema},
            {
                "role": "user",
                "content": (
                    "Run the lint workflow on the wiki index below. "
                    "Return a markdown lint report. Do NOT call write_wiki_page.\n\n"
                    f"## Wiki index\n\n{index_text}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    report = response.choices[0].message.content or "No issues found."

    await db.table("wiki_log").insert({
        "event_type": "lint",
        "summary": f"Lint run: {len(index_entries)} pages checked.",
    }).execute()

    return report
