"""
Ingestion endpoints.

POST /api/ingest/upload
  Accepts a CSV or XLSX file, runs Stages 1–3 synchronously (parse,
  classify stub, store), then creates an ingestion_jobs row for the
  worker to complete Stages 4–6.

GET /api/ingest/status/{job_id}
  Returns the current stage, status, and progress percentage of a job.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.database import get_db
from app.models.jobs import JobStatusOut, UploadOut
from app.models.surveys import DatasetOut, ParsedSurvey
from app.services.parser import infer_respondent_ref, infer_responded_at, parse_survey_file

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _ext(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/upload", response_model=UploadOut)
async def upload_survey(
    file: UploadFile = File(...),
    survey_name: str | None = Form(None),
    survey_type: str = Form("participant"),
    conducted_at: str | None = Form(None),
    db: Any = Depends(get_db),
) -> UploadOut:
    """
    Upload a survey file. Stages 1–3 run synchronously so the user gets
    immediate confirmation. Stages 4–6 run in the background worker.
    """
    filename = file.filename or "upload.csv"

    if _ext(filename) not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")

    # ── Stage 1: Parse ────────────────────────────────────────────────────────
    try:
        parsed = parse_survey_file(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    name = survey_name or parsed.name
    survey_id = str(uuid.uuid4())

    # ── Stage 1: Write survey row ─────────────────────────────────────────────
    survey_row = {
        "id": survey_id,
        "name": name,
        "type": survey_type,
        "source": parsed.source,
        "conducted_at": conducted_at,
        "row_count": parsed.row_count,
        "column_count": parsed.column_count,
        "file_name": filename,
    }
    await db.table("surveys").insert(survey_row).execute()
    log.info("Created survey %s: %s (%d rows)", survey_id, name, parsed.row_count)

    # ── Stage 2: Write survey_questions ──────────────────────────────────────
    # question_type defaults to 'open_ended'; the worker classifies properly.
    question_rows = [
        {
            "survey_id": survey_id,
            "column_key": q["column_key"],
            "label": q["label"],
            "question_type": "open_ended",  # overwritten by classify stage
            "position": q["position"],
        }
        for q in parsed.questions
    ]
    if question_rows:
        await db.table("survey_questions").insert(question_rows).execute()

    # ── Stage 3: Write survey_responses + open_ended_answers ─────────────────
    # Fetch question IDs so we can link open-ended answers.
    q_result = (
        await db.table("survey_questions")
        .select("id, column_key, position")
        .eq("survey_id", survey_id)
        .execute()
    )
    col_to_id = {q["column_key"]: q["id"] for q in (q_result.data or [])}

    response_rows = []
    oe_rows = []

    for row in parsed.rows:
        response_id = str(uuid.uuid4())
        structured = {
            col_to_id[k]: v
            for k, v in row.items()
            if k in col_to_id and v is not None and str(v) != "nan"
        }
        response_rows.append({
            "id": response_id,
            "survey_id": survey_id,
            "respondent_ref": infer_respondent_ref(row),
            "responded_at": infer_responded_at(row),
            "structured": structured,
        })

        # Collect open-ended text for all columns (classifier will sort types later).
        for col_key, q_id in col_to_id.items():
            val = row.get(col_key)
            if val is None or str(val).strip() in ("", "nan"):
                continue
            text = str(val).strip()
            if len(text) > 10:  # Ignore very short answers.
                oe_rows.append({
                    "response_id": response_id,
                    "question_id": q_id,
                    "answer_text": text,
                })

    # Batch insert to stay within Supabase's default request size.
    _batch = 500
    for i in range(0, len(response_rows), _batch):
        await db.table("survey_responses").insert(response_rows[i : i + _batch]).execute()
    for i in range(0, len(oe_rows), _batch):
        await db.table("open_ended_answers").insert(oe_rows[i : i + _batch]).execute()

    log.info("Stored %d responses, %d open-ended answers.", len(response_rows), len(oe_rows))

    # ── Create ingestion job for worker (Stages 4–6) ─────────────────────────
    job_id = str(uuid.uuid4())
    await db.table("ingestion_jobs").insert({
        "id": job_id,
        "survey_id": survey_id,
        "stage": "classify",  # worker picks up from here
        "status": "pending",
        "attempt": 0,
    }).execute()

    log.info("Created ingestion job %s for survey %s.", job_id, survey_id)
    return UploadOut(jobId=job_id, surveyId=survey_id)


@router.get("/status/{job_id}", response_model=JobStatusOut)
async def get_job_status(job_id: str, db: Any = Depends(get_db)) -> JobStatusOut:
    """Poll ingestion job progress."""
    result = (
        await db.table("ingestion_jobs")
        .select("*")
        .eq("id", job_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusOut.from_job(result.data)
