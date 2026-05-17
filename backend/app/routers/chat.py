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
from app.services.query_agent import stream_answer

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


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
    SSE generator. Events emitted:
      status  — brief progress message while retrieving context
      chunk   — one token from the LLM (real-time Groq streaming)
      sources — list of source objects after the answer is complete
      done    — signals end of stream
      error   — if something goes wrong
    """
    try:
        yield _SSE.event("status", {"message": "Searching knowledge base..."})
        await asyncio.sleep(0)

        answer_parts: list[str] = []
        sources: list = []

        async for event in stream_answer(
            db=db,
            question=message,
            mode=mode,
            dataset_ids=dataset_ids or None,
        ):
            if event["type"] == "chunk":
                answer_parts.append(event["text"])
                yield _SSE.event("chunk", {"text": event["text"]})
            elif event["type"] == "done":
                sources = event.get("sources", [])

        answer = "".join(answer_parts)

        # Load or create the session, append the two new messages, save.
        session = await _load_session(db, session_id)
        if session is None:
            session = {
                "id": session_id,
                "title": message[:60],
                "messages": [],
                "mode": mode,
                "createdAt": _now_iso(),
            }
        session["messages"].extend([
            {"id": str(uuid.uuid4()), "role": "user",
             "content": message, "timestamp": _now_iso()},
            {"id": str(uuid.uuid4()), "role": "assistant",
             "content": answer, "timestamp": _now_iso(), "sources": sources},
        ])
        await _save_session(db, session)

        yield _SSE.event("sources", sources)
        yield _SSE.event("done", {"sessionId": session_id})

    except Exception as exc:
        log.exception("Chat stream error:")
        yield _SSE.event("error", {"message": str(exc)})


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


async def _load_session(db: Any, session_id: str) -> dict | None:
    try:
        result = (
            await db.table("chat_sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        return result.data or None
    except Exception:
        return None


async def _save_session(db: Any, session: dict) -> None:
    try:
        await db.table("chat_sessions").upsert({
            "id":       session["id"],
            "title":    session.get("title", "")[:120],
            "mode":     session.get("mode", "standard"),
            "messages": session.get("messages", []),
        }).execute()
    except Exception as exc:
        log.warning("Failed to save session %s: %s", session.get("id"), exc)


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
async def list_sessions(db: Any = Depends(get_db)) -> list[dict]:
    result = (
        await db.table("chat_sessions")
        .select("id, title, mode, created_at, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


@router.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str, db: Any = Depends(get_db)) -> dict:
    session = await _load_session(db, session_id)
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
