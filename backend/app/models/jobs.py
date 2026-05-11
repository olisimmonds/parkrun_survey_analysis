from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

STAGES = ["parse", "classify", "store", "embed", "cluster", "wiki_update", "done"]

STAGE_PROGRESS: dict[str, int] = {
    "parse": 10,
    "classify": 20,
    "store": 35,
    "embed": 55,
    "cluster": 75,
    "wiki_update": 90,
    "done": 100,
}


class JobCreate(BaseModel):
    survey_id: UUID
    stage: str = "parse"
    status: str = "pending"


class Job(BaseModel):
    id: UUID
    survey_id: UUID
    stage: str
    status: str
    attempt: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobStatusOut(BaseModel):
    """API response for GET /api/ingest/status/:jobId"""
    jobId: str
    surveyId: str
    stage: str
    status: str
    progress: int
    error: str | None = None

    @classmethod
    def from_job(cls, job: dict) -> "JobStatusOut":
        return cls(
            jobId=str(job["id"]),
            surveyId=str(job["survey_id"]),
            stage=job["stage"],
            status=job["status"],
            progress=STAGE_PROGRESS.get(job["stage"], 0),
            error=job.get("last_error"),
        )


class UploadOut(BaseModel):
    """API response for POST /api/ingest/upload"""
    jobId: str
    surveyId: str
    message: str = "Upload received. Processing has started."
