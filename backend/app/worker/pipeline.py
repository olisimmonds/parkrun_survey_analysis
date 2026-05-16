"""
Ingestion pipeline worker.

Polls the ingestion_jobs table for pending work and advances jobs through
the six processing stages:

  parse → classify → store → embed → cluster → wiki_update → done

Each stage is atomic: the job is marked 'running' before work begins and
advanced to the next stage (or marked 'failed') when it ends. If the
worker crashes mid-stage, the job is left in 'running' and a watchdog
reset will restart it (see reset_stale_jobs).

Run the worker:
  python -m app.worker.pipeline

Or integrated with the FastAPI server for local development:
  python -m app.main --with-worker
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.services import classifier, clusterer, embedder, parser, wiki_maintainer

log = logging.getLogger(__name__)


async def _get_db():
    """Create a fresh async Supabase client for the worker process."""
    from supabase._async.client import create_client  # type: ignore

    settings = get_settings()
    return await create_client(settings.supabase_url, settings.supabase_service_key)


# ─── Stage implementations ────────────────────────────────────────────────────


async def _stage_classify(db: Any, job: dict) -> None:
    """
    Stage 2: Use Groq to classify each survey question's type.
    Reads survey_questions for this survey, classifies in one batch call,
    and writes question_type back to each row.
    """
    survey_id = job["survey_id"]

    questions_result = (
        await db.table("survey_questions")
        .select("id, column_key, label, position")
        .eq("survey_id", survey_id)
        .order("position")
        .execute()
    )
    questions = questions_result.data or []
    if not questions:
        log.warning("No questions found for survey %s — skipping classify.", survey_id)
        return

    # Fetch distinct values per column to help the classifier.
    responses_result = (
        await db.table("survey_responses")
        .select("structured")
        .eq("survey_id", survey_id)
        .limit(500)
        .execute()
    )
    rows = [r["structured"] for r in (responses_result.data or [])]
    distinct = classifier.compute_distinct_values(rows)

    classified = await classifier.classify_questions(questions, distinct)

    for q in classified:
        await (
            db.table("survey_questions")
            .update({"question_type": q["question_type"]})
            .eq("id", q["id"])
            .execute()
        )

    log.info("Classified %d questions for survey %s.", len(classified), survey_id)


async def _stage_embed(db: Any, job: dict) -> None:
    """
    Stage 4: Compute 768-dim embeddings for all open-ended answers in this survey.
    Processes in batches and writes vectors to open_ended_answers.embedding.
    """
    survey_id = job["survey_id"]

    answers_result = (
        await db.table("open_ended_answers")
        .select("id, answer_text")
        .is_("embedding", "null")
        .eq(
            "question_id",
            # Sub-select: question_ids for this survey.
            db.table("survey_questions")
            .select("id")
            .eq("survey_id", survey_id),
        )
        .execute()
    )
    answers = answers_result.data or []

    if not answers:
        log.info("No un-embedded answers for survey %s.", survey_id)
        return

    texts = [a["answer_text"] for a in answers]
    log.info("Embedding %d answers for survey %s.", len(texts), survey_id)

    vectors = await embedder.embed_documents(texts)

    # Write back in batches of 100 to stay within Supabase request limits.
    batch_size = 100
    for i in range(0, len(answers), batch_size):
        batch = answers[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        updates = [
            {"id": a["id"], "embedding": v}
            for a, v in zip(batch, batch_vecs)
        ]
        await db.table("open_ended_answers").upsert(updates).execute()

    log.info("Wrote embeddings for survey %s.", survey_id)


async def _stage_embed_v2(db: Any, job: dict) -> None:
    """
    Simpler embed implementation that fetches answers via question join.
    """
    survey_id = job["survey_id"]

    # Get open-ended question IDs for this survey.
    q_result = (
        await db.table("survey_questions")
        .select("id")
        .eq("survey_id", survey_id)
        .eq("question_type", "open_ended")
        .execute()
    )
    q_ids = [q["id"] for q in (q_result.data or [])]
    if not q_ids:
        log.info("No open-ended questions for survey %s — skipping embed.", survey_id)
        return

    # Fetch answers without embeddings.
    answers_result = (
        await db.table("open_ended_answers")
        .select("id, answer_text")
        .in_("question_id", q_ids)
        .is_("embedding", "null")
        .execute()
    )
    answers = answers_result.data or []
    if not answers:
        log.info("No un-embedded answers for survey %s.", survey_id)
        return

    texts = [a["answer_text"] for a in answers]
    log.info("Embedding %d answers for survey %s.", len(texts), survey_id)
    vectors = await embedder.embed_documents(texts)

    # Use UPDATE (not upsert) — upsert's INSERT path fails the NOT NULL constraint
    # on response_id/question_id even with default_to_null=False. Run concurrently
    # in batches of 50 to avoid overwhelming the connection pool.
    async def _write_embedding(answer_id: str, vec: list) -> None:
        await (
            db.table("open_ended_answers")
            .update({"embedding": vec})
            .eq("id", answer_id)
            .execute()
        )

    batch_size = 50
    for i in range(0, len(answers), batch_size):
        batch = answers[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        await asyncio.gather(*[_write_embedding(a["id"], v) for a, v in zip(batch, batch_vecs)])
        log.debug("Wrote embedding batch %d–%d.", i, i + len(batch))

    log.info("Wrote embeddings for survey %s.", survey_id)


async def _stage_cluster(db: Any, job: dict) -> None:
    """
    Stage 5: HDBSCAN clustering of open-ended answers per question.
    Calls the LLM to label each cluster, then writes response_clusters rows.
    """
    survey_id = job["survey_id"]

    q_result = (
        await db.table("survey_questions")
        .select("id, label")
        .eq("survey_id", survey_id)
        .eq("question_type", "open_ended")
        .execute()
    )
    questions = q_result.data or []

    for question in questions:
        q_id = question["id"]
        q_label = question["label"]

        answers_result = (
            await db.table("open_ended_answers")
            .select("id, answer_text, embedding")
            .eq("question_id", q_id)
            .not_.is_("embedding", "null")
            .execute()
        )
        answers = answers_result.data or []
        if len(answers) < 3:
            log.info("Too few answers (%d) for question %s — skipping.", len(answers), q_id)
            continue

        # Supabase/PostgREST returns pgvector values as JSON strings; parse them.
        def _parse_vec(v):
            if isinstance(v, str):
                import json
                return json.loads(v)
            return v

        embeddings = [_parse_vec(a["embedding"]) for a in answers]
        answer_ids = [a["id"] for a in answers]
        texts = [a["answer_text"] for a in answers]

        raw_clusters = clusterer.cluster_embeddings(embeddings, answer_ids, texts)
        if not raw_clusters:
            continue

        labeled = await clusterer.label_clusters(raw_clusters, q_label)

        # Bulk-upsert all cluster rows for this question.
        cluster_rows = []
        theme_updates: list[dict] = []
        for cl in labeled:
            rep_quotes = [
                {"text": t, "answer_id": aid}
                for t, aid in zip(cl["representative_texts"], cl["representative_ids"])
            ]
            cluster_rows.append({
                "survey_id": str(survey_id),
                "question_id": q_id,
                "cluster_id": cl["cluster_id"],
                "label": cl.get("label", ""),
                "summary": cl.get("summary", ""),
                "response_count": len(cl["member_ids"]),
                "centroid": cl["centroid"],
                "representative_quotes": rep_quotes,
            })
            for answer_id in cl["member_ids"]:
                theme_updates.append({"id": answer_id, "theme_cluster": cl["cluster_id"]})

        if cluster_rows:
            await db.table("response_clusters").upsert(cluster_rows).execute()

        # Concurrent UPDATE (not upsert) for the same reason as the embed stage.
        async def _write_theme(answer_id: str, cluster_id: int) -> None:
            await (
                db.table("open_ended_answers")
                .update({"theme_cluster": cluster_id})
                .eq("id", answer_id)
                .execute()
            )

        theme_batch = 50
        for i in range(0, len(theme_updates), theme_batch):
            batch = theme_updates[i : i + theme_batch]
            await asyncio.gather(*[_write_theme(u["id"], u["theme_cluster"]) for u in batch])

        log.info("Clustered question '%s': %d clusters.", q_label, len(labeled))


async def _stage_wiki_update(db: Any, job: dict) -> None:
    """
    Stage 6: Run the LLMWiki ingest for this survey.
    Materialises cluster summaries and calls the wiki maintainer.
    """
    survey_id = job["survey_id"]

    survey_result = (
        await db.table("surveys")
        .select("name, type, conducted_at, row_count")
        .eq("id", survey_id)
        .single()
        .execute()
    )
    survey = survey_result.data
    if not survey:
        log.error("Survey %s not found for wiki update.", survey_id)
        return

    # Gather clusters with question labels.
    clusters_result = (
        await db.table("response_clusters")
        .select("label, summary, response_count, representative_quotes, question_id")
        .eq("survey_id", survey_id)
        .execute()
    )
    raw_clusters = clusters_result.data or []

    # Enrich with question labels.
    clusters_for_wiki = []
    for cl in raw_clusters:
        q_result = (
            await db.table("survey_questions")
            .select("label")
            .eq("id", cl["question_id"])
            .single()
            .execute()
        )
        q_label = (q_result.data or {}).get("label", "")
        clusters_for_wiki.append({
            "question_label": q_label,
            "label": cl.get("label", ""),
            "summary": cl.get("summary", ""),
            "response_count": cl.get("response_count", 0),
            "representative_quotes": cl.get("representative_quotes", []),
        })

    conducted_at = survey.get("conducted_at")
    if conducted_at:
        conducted_at = str(conducted_at)[:10]  # ISO date

    await wiki_maintainer.run_ingest(
        db=db,
        survey_id=UUID(survey_id) if isinstance(survey_id, str) else survey_id,
        survey_name=survey["name"],
        survey_type=survey["type"],
        conducted_at=conducted_at,
        row_count=survey.get("row_count") or 0,
        clusters=clusters_for_wiki,
        sql_stats={},  # Full SQL stats added in Phase 2.
    )


# ─── Stage registry ────────────────────────────────────────────────────────────

_NEXT_STAGE = {
    "parse": "classify",
    "classify": "store",
    "store": "embed",
    "embed": "cluster",
    "cluster": "wiki_update",
    "wiki_update": "done",
}

_STAGE_FN = {
    "classify": _stage_classify,
    "embed": _stage_embed_v2,
    "cluster": _stage_cluster,
    "wiki_update": _stage_wiki_update,
}


async def _advance_job(db: Any, job: dict, next_stage: str) -> None:
    await (
        db.table("ingestion_jobs")
        .update({"stage": next_stage, "status": "pending" if next_stage != "done" else "done"})
        .eq("id", job["id"])
        .execute()
    )


async def _fail_job(db: Any, job: dict, error: str) -> None:
    attempt = job.get("attempt", 0) + 1
    settings = get_settings()
    if attempt >= settings.max_job_retries:
        await (
            db.table("ingestion_jobs")
            .update({"status": "failed", "last_error": error, "attempt": attempt})
            .eq("id", job["id"])
            .execute()
        )
        log.error("Job %s permanently failed after %d attempts: %s", job["id"], attempt, error)
    else:
        await (
            db.table("ingestion_jobs")
            .update({"status": "pending", "last_error": error, "attempt": attempt})
            .eq("id", job["id"])
            .execute()
        )
        log.warning("Job %s failed (attempt %d/%d): %s", job["id"], attempt, settings.max_job_retries, error)


async def process_one(db: Any, job: dict) -> None:
    """Process a single pending job through its current stage."""
    stage = job["stage"]
    job_id = job["id"]

    # 'parse' and 'store' run inside the upload endpoint, not the worker.
    # If a job arrives at either stage, just advance it — no function to call.
    if stage not in _STAGE_FN and stage not in ("parse", "store"):
        log.warning("Unknown stage '%s' on job %s — skipping.", stage, job_id)
        return

    # Mark running.
    await (
        db.table("ingestion_jobs")
        .update({"status": "running"})
        .eq("id", job_id)
        .execute()
    )

    try:
        if stage in _STAGE_FN:
            await _STAGE_FN[stage](db, job)
        # 'parse' and 'store' stages are executed synchronously in the upload
        # endpoint before the job is handed to the worker — the worker picks up
        # from 'classify' onwards. If parse/store appear here it means a restart
        # occurred mid-ingest; they are safe to re-run.

        next_stage = _NEXT_STAGE.get(stage, "done")
        await _advance_job(db, job, next_stage)
        log.info("Job %s: %s → %s", job_id, stage, next_stage)

    except Exception as exc:
        log.exception("Job %s stage '%s' raised:", job_id, stage)
        await _fail_job(db, job, str(exc))


async def reset_stale_jobs(db: Any, stale_minutes: int = 30) -> int:
    """
    Reset jobs stuck in 'running' state (e.g. from a previous crashed worker).
    Returns the number of jobs reset.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()
    result = (
        await db.table("ingestion_jobs")
        .update({"status": "pending", "last_error": "Reset after stale timeout"})
        .eq("status", "running")
        .lt("updated_at", cutoff)
        .execute()
    )
    n = len(result.data or [])
    if n:
        log.warning("Reset %d stale jobs.", n)
    return n


async def _nightly_lint_loop(db: Any) -> None:
    """
    Runs wiki_maintainer.run_lint once per day at midnight UTC.
    Runs as a background asyncio task alongside the main worker loop.
    """
    from datetime import datetime, timedelta, timezone

    while True:
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wait_seconds = (next_midnight - now).total_seconds()
        log.info("Wiki lint scheduled in %.0f s (next midnight UTC).", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            report = await wiki_maintainer.run_lint(db)
            log.info("Nightly wiki lint complete.\n%s", report)
        except Exception:
            log.exception("Nightly wiki lint failed:")


async def run_worker() -> None:
    """Main worker loop. Polls the job queue and processes pending jobs."""
    settings = get_settings()
    log.info("Worker starting — poll interval: %ds", settings.worker_poll_interval_seconds)

    db = await _get_db()
    await reset_stale_jobs(db)

    # Start nightly lint as a background task.
    asyncio.create_task(_nightly_lint_loop(db))

    while True:
        try:
            result = (
                await db.table("ingestion_jobs")
                .select("*")
                .eq("status", "pending")
                .neq("stage", "done")
                .order("created_at")
                .limit(1)
                .execute()
            )
            jobs = result.data or []
            if jobs:
                await process_one(db, jobs[0])
            else:
                await asyncio.sleep(settings.worker_poll_interval_seconds)
        except Exception:
            log.exception("Worker loop error — continuing after short delay.")
            await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run_worker())
