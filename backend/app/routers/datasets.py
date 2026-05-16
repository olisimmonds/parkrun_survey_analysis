"""
Dataset management endpoints.

GET /api/datasets         — list all surveys
GET /api/datasets/{id}    — get a single survey
DELETE /api/datasets/{id} — delete a survey and all its data
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.models.surveys import DatasetOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _survey_to_dataset(survey: dict) -> DatasetOut:
    """Map a surveys row to the DatasetOut shape expected by the frontend."""
    return DatasetOut(
        id=str(survey["id"]),
        name=survey["name"],
        type=survey["type"].capitalize(),
        uploadedAt=survey.get("uploaded_at", ""),
        size=0,  # File size not stored server-side; not shown in the UI.
        rowCount=survey.get("row_count"),
        columnCount=survey.get("column_count"),
        description="",
        tags=[],
        status=_infer_status(survey),
        fileName=survey.get("file_name", ""),
    )


def _infer_status(survey: dict) -> str:
    """Derive a DatasetStatus from the most recent ingestion job for this survey."""
    job_status = survey.get("_job_status")
    if job_status == "failed":
        return "error"
    if job_status in ("pending", "running"):
        return "processing"
    return "ready"


@router.get("", response_model=list[DatasetOut])
async def list_datasets(db: Any = Depends(get_db)) -> list[DatasetOut]:
    """Return all surveys ordered by upload date (newest first)."""
    result = (
        await db.table("surveys")
        .select("*")
        .order("uploaded_at", desc=True)
        .execute()
    )
    surveys = result.data or []

    # Annotate each survey with the status of its latest ingestion job.
    for survey in surveys:
        job_result = (
            await db.table("ingestion_jobs")
            .select("status")
            .eq("survey_id", survey["id"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        jobs = job_result.data or []
        survey["_job_status"] = jobs[0]["status"] if jobs else "done"

    return [_survey_to_dataset(s) for s in surveys]


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(dataset_id: str, db: Any = Depends(get_db)) -> DatasetOut:
    result = (
        await db.table("surveys").select("*").eq("id", dataset_id).single().execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    survey = result.data
    job_result = (
        await db.table("ingestion_jobs")
        .select("status")
        .eq("survey_id", dataset_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    jobs = job_result.data or []
    survey["_job_status"] = jobs[0]["status"] if jobs else "done"

    return _survey_to_dataset(survey)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, db: Any = Depends(get_db)) -> None:
    """
    Delete a survey and all associated data, including wiki pages and log entries.

    ON DELETE CASCADE handles: survey_questions, survey_responses, open_ended_answers,
    response_clusters, ingestion_jobs. Wiki tables use a survey_ids array (no FK), so
    we clean those up manually before deleting the survey row.
    """
    result = await db.table("surveys").select("id").eq("id", dataset_id).execute()
    if not (result.data or []):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Clean up wiki_pages that reference this survey.
    wiki_result = await db.table("wiki_pages").select("id, survey_ids").execute()
    for page in (wiki_result.data or []):
        ids = [str(i) for i in (page.get("survey_ids") or [])]
        if dataset_id not in ids:
            continue
        remaining = [i for i in ids if i != dataset_id]
        if not remaining:
            await db.table("wiki_pages").delete().eq("id", page["id"]).execute()
        else:
            await db.table("wiki_pages").update({"survey_ids": remaining}).eq("id", page["id"]).execute()

    # Clean up wiki_log entries for this survey.
    await db.table("wiki_log").delete().eq("survey_id", dataset_id).execute()

    # Delete the survey row — cascade handles everything else.
    await db.table("surveys").delete().eq("id", dataset_id).execute()
    log.info("Deleted survey %s and all associated data.", dataset_id)
