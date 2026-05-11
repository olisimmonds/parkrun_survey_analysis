"""
Chat endpoint with Server-Sent Events streaming.

POST /api/chat
  Accepts a user message and QueryConfig. Routes to the query agent,
  streams tool status updates as SSE events, then streams the final answer.

GET /api/chat/sessions
GET /api/chat/sessions/{session_id}
  Session retrieval (Phase 2 — sessions are stored in-memory for MVP).

GET /api/wiki
GET /api/wiki/{slug}
  Browse the wiki knowledge base.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.database import get_db
from app.models.wiki import WikiIndexEntry, WikiIndexOut
from app.services.query_agent import answer_question

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# In-memory session store for MVP.
_sessions: dict[str, dict] = {}


class _SSE:
    """Tiny SSE helper for consistent event formatting."""

    @staticmethod
    def event(event_type: str, data: Any) -> str:
        payload = json.dumps(data) if not isinstance(data, str) else data
        return f"event: {event_type}\ndata: {payload}\n\n"


async def _stream_chat(
    message: str,
    mode: str,
    dataset_ids: list[str],
    session_id: str,
    db: Any,
):
    """
    Generator that yields SSE-formatted events:
      status   — tool is running ("Searching wiki...", "Running SQL...", etc.)
      chunk    — a word/token of the final answer (simulated streaming)
      sources  — list of source objects
      done     — signals end of stream
      error    — if something goes wrong
    """
    try:
        yield _SSE.event("status", {"message": "Analysing question…"})
        await asyncio.sleep(0)

        yield _SSE.event("status", {"message": "Searching wiki knowledge base…"})
        await asyncio.sleep(0)

        result = await answer_question(
            db=db,
            question=message,
            mode=mode,
            dataset_ids=dataset_ids or None,
        )

        answer: str = result["answer"]
        sources = result.get("sources", [])

        yield _SSE.event("status", {"message": "Composing response…"})

        # Stream the answer word by word (real token streaming requires an async
        # generator from the LLM; Groq streaming can be added in the next phase).
        words = answer.split(" ")
        for word in words:
            yield _SSE.event("chunk", {"text": word + " "})
            await asyncio.sleep(0.01)

        # Store message in session.
        if session_id not in _sessions:
            _sessions[session_id] = {
                "id": session_id,
                "title": message[:60],
                "messages": [],
                "mode": mode,
                "createdAt": _now_iso(),
            }
        _sessions[session_id]["messages"].append(
            {"id": str(uuid.uuid4()), "role": "user", "content": message, "timestamp": _now_iso()}
        )
        _sessions[session_id]["messages"].append(
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": answer,
                "timestamp": _now_iso(),
                "sources": sources,
            }
        )

        yield _SSE.event("sources", sources)
        yield _SSE.event("done", {"sessionId": session_id})

    except Exception as exc:
        log.exception("Chat stream error:")
        yield _SSE.event("error", {"message": str(exc)})


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@router.post("/api/chat")
async def chat(
    body: dict,
    db: Any = Depends(get_db),
) -> StreamingResponse:
    """
    Stream an AI response for a user message.

    Request body:
      {
        "message": str,
        "config": { "mode": "standard"|"deep-research", "datasetIds": [], "maxSources": 5 },
        "sessionId": str  (optional)
      }
    """
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message is required.")

    config = body.get("config", {})
    mode = config.get("mode", "standard")
    dataset_ids = config.get("datasetIds", [])
    session_id = body.get("sessionId") or str(uuid.uuid4())

    return StreamingResponse(
        _stream_chat(message, mode, dataset_ids, session_id, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE.
        },
    )


@router.get("/api/chat/sessions")
async def list_sessions() -> list[dict]:
    return sorted(_sessions.values(), key=lambda s: s["createdAt"], reverse=True)


@router.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.get("/api/wiki", response_model=WikiIndexOut)
async def get_wiki_index(db: Any = Depends(get_db)) -> WikiIndexOut:
    result = (
        await db.table("wiki_pages")
        .select("slug, title, page_type, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    pages = result.data or []
    return WikiIndexOut(
        pages=[
            WikiIndexEntry(
                slug=p["slug"],
                title=p["title"],
                page_type=p["page_type"],
                updated_at=str(p.get("updated_at", ""))[:10],
            )
            for p in pages
        ],
        total=len(pages),
    )


@router.get("/api/wiki/{slug:path}")
async def get_wiki_page(slug: str, db: Any = Depends(get_db)) -> dict:
    result = (
        await db.table("wiki_pages")
        .select("slug, page_type, title, content, linked_slugs, survey_ids, updated_at")
        .eq("slug", slug)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Wiki page '{slug}' not found.")
    return result.data
