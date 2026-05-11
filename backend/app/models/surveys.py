from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SurveyCreate(BaseModel):
    name: str
    type: str
    source: str | None = None
    conducted_at: datetime | None = None
    row_count: int | None = None
    column_count: int | None = None
    file_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Survey(SurveyCreate):
    id: UUID
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class QuestionCreate(BaseModel):
    survey_id: UUID
    column_key: str
    label: str
    question_type: str
    position: int
    options: list[str] | None = None


class Question(QuestionCreate):
    id: UUID

    model_config = {"from_attributes": True}


class ResponseCreate(BaseModel):
    survey_id: UUID
    respondent_ref: str | None = None
    responded_at: datetime | None = None
    structured: dict[str, Any] = Field(default_factory=dict)


class OpenEndedAnswerCreate(BaseModel):
    response_id: UUID
    question_id: UUID
    answer_text: str


# API response shapes --------------------------------------------------------

class DatasetOut(BaseModel):
    """Matches the Dataset type expected by the Next.js frontend."""
    id: str
    name: str
    type: str
    uploadedAt: str
    size: int
    rowCount: int | None = None
    columnCount: int | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    status: str = "ready"
    fileName: str


class ParsedSurvey(BaseModel):
    """Intermediate in-memory structure produced by the CSV parser."""
    name: str
    source: str
    questions: list[dict[str, Any]]   # [{column_key, label, position}]
    rows: list[dict[str, Any]]         # raw data rows keyed by column_key
    row_count: int
    column_count: int
    file_name: str
