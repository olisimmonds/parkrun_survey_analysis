from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WikiPageCreate(BaseModel):
    slug: str
    page_type: str
    title: str
    content: str
    survey_ids: list[UUID] = Field(default_factory=list)
    linked_slugs: list[str] = Field(default_factory=list)


class WikiPage(WikiPageCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WikiIndexEntry(BaseModel):
    """One-line entry in the wiki index — sent to the LLM without full page content."""
    slug: str
    title: str
    page_type: str
    updated_at: str  # ISO date string


class WikiIndexOut(BaseModel):
    pages: list[WikiIndexEntry]
    total: int


class WikiLogCreate(BaseModel):
    event_type: str
    page_slug: str | None = None
    survey_id: UUID | None = None
    summary: str | None = None
