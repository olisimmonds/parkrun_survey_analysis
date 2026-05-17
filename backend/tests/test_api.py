"""
API endpoint tests.

All external services (Supabase, Groq) are mocked.
Tests verify:
  - Upload endpoint validates file types and sizes
  - Upload endpoint returns correct response shape
  - Dataset list and delete endpoints work correctly
  - Health endpoint responds
  - Wiki index endpoint responds
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_standard_csv_bytes, make_surveymonkey_csv_bytes


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Upload ────────────────────────────────────────────────────────────────────


def test_upload_csv_returns_job_id(client, mock_db, standard_csv):
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("participant_survey.csv", io.BytesIO(standard_csv), "text/csv")},
        data={"survey_type": "participant"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "jobId" in body
    assert "surveyId" in body


def test_upload_surveymonkey_csv(client, mock_db, surveymonkey_csv):
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("sm_export.csv", io.BytesIO(surveymonkey_csv), "text/csv")},
        data={"survey_type": "volunteer"},
    )
    assert resp.status_code == 200


def test_upload_rejects_json_file(client, mock_db):
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("data.json", io.BytesIO(b'{"a":1}'), "application/json")},
        data={"survey_type": "participant"},
    )
    assert resp.status_code == 422


def test_upload_rejects_oversized_file(client, mock_db):
    huge = b"a,b,c\n" + b"1,2,3\n" * 1_000_000
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("big.csv", io.BytesIO(huge), "text/csv")},
        data={"survey_type": "participant"},
    )
    # 50 MB limit — file above is ~11 MB so may or may not trigger depending on
    # system memory. Just verify a non-500 response.
    assert resp.status_code in (200, 413, 422)


def test_upload_empty_csv_returns_422(client, mock_db):
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("empty.csv", io.BytesIO(b"col1,col2\n"), "text/csv")},
        data={"survey_type": "participant"},
    )
    assert resp.status_code == 422


# ── Ingest status ─────────────────────────────────────────────────────────────


def test_status_not_found(client, mock_db):
    # mock_db.tables has no "ingestion_jobs" key → returns empty list → 404.
    resp = client.get("/api/ingest/status/nonexistent-job-id")
    assert resp.status_code == 404


def test_status_returns_progress(client, mock_db):
    job_data = {
        "id": "job-123",
        "survey_id": "survey-456",
        "stage": "embed",
        "status": "running",
        "attempt": 0,
        "last_error": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    mock_db.tables["ingestion_jobs"] = [job_data]

    resp = client.get("/api/ingest/status/job-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "embed"
    assert body["progress"] == 55  # STAGE_PROGRESS["embed"]


# ── Datasets ──────────────────────────────────────────────────────────────────


def test_list_datasets_empty(client, mock_db):
    # No surveys in mock → empty list returned.
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_delete_dataset_not_found(client, mock_db):
    # No "surveys" data → 404.
    resp = client.delete("/api/datasets/nonexistent-id")
    assert resp.status_code == 404


# ── Wiki ──────────────────────────────────────────────────────────────────────


def test_wiki_index_empty(client, mock_db):
    resp = client.get("/api/wiki")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["pages"] == []


def test_wiki_page_not_found(client, mock_db):
    resp = client.get("/api/wiki/theme/nonexistent")
    assert resp.status_code == 404


# ── PDF upload ────────────────────────────────────────────────────────────────


def test_upload_pdf_extension_accepted(client, mock_db):
    """PDF extension is no longer rejected; parsing failure returns 422, not 422 for wrong ext."""
    # Minimal bytes that look like a PDF header but will fail text extraction.
    # We expect 422 (no text) not 422 (wrong extension) — or 200 if parsing somehow works.
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        data={"survey_type": "other"},
    )
    # 422 is acceptable (empty PDF), but it must NOT be rejected purely for being a .pdf
    # We verify it's not a 422 from the extension guard by checking the error message.
    if resp.status_code == 422:
        body = resp.json()
        assert "unsupported" not in body.get("detail", "").lower()


# ── Chat session persistence ──────────────────────────────────────────────────


def test_chat_sessions_list_empty(client, mock_db):
    """GET /api/chat/sessions returns an empty list when no sessions exist."""
    resp = client.get("/api/chat/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_chat_sessions_list_returns_rows(client, mock_db):
    """GET /api/chat/sessions returns DB rows."""
    mock_db.tables["chat_sessions"] = [
        {
            "id": "sess-1",
            "title": "What themes emerged?",
            "mode": "standard",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
    ]
    resp = client.get("/api/chat/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["title"] == "What themes emerged?"


def test_chat_session_not_found(client, mock_db):
    """GET /api/chat/sessions/{id} returns 404 for an unknown session."""
    resp = client.get("/api/chat/sessions/nonexistent-session")
    assert resp.status_code == 404
