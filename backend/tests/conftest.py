"""
Shared pytest fixtures.

The test suite runs entirely without a live Supabase connection or API keys.
External calls (Groq, Supabase) are patched at the module level.
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


# ── File fixtures ──────────────────────────────────────────────────────────────


def make_csv_bytes(rows: list[dict]) -> bytes:
    """Create in-memory CSV bytes from a list of dicts."""
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_surveymonkey_csv_bytes() -> bytes:
    """Create a SurveyMonkey-style two-row-header CSV."""
    lines = [
        "Background,Background,Background,Experience,Experience,Experience",
        "Age group,Gender,Home parkrun,Overall satisfaction,What did you enjoy?,Suggestions",
        "25-34,Female,Southampton,5,Great community feel,More parking",
        "35-44,Male,Winchester,4,Friendly volunteers,Better signage",
        "18-24,Female,Basingstoke,5,Inclusive atmosphere,Online results faster",
        "45-54,Male,Southampton,3,Regular routine,Shorter queue at finish",
        "55-64,Female,Portsmouth,5,Meeting people,Nothing to improve",
    ]
    return "\n".join(lines).encode()


def make_standard_csv_bytes() -> bytes:
    """Create a standard single-header CSV (Google Forms style)."""
    rows = [
        {
            "RespondentID": 1001,
            "StartDate": "2026-02-01",
            "AgeGroup": "25-34",
            "Gender": "Male",
            "Home_parkrun": "Southampton",
            "How_often": "Every week",
            "What_enjoy": "Community atmosphere and friendly volunteers",
            "Improvements": "More parking near the start",
            "Recommend": "Yes",
        },
        {
            "RespondentID": 1002,
            "StartDate": "2026-02-01",
            "AgeGroup": "35-44",
            "Gender": "Female",
            "Home_parkrun": "Winchester",
            "How_often": "Most weeks",
            "What_enjoy": "The inclusive nature and seeing people of all abilities",
            "Improvements": "Maybe clearer signage for first timers",
            "Recommend": "Yes",
        },
        {
            "RespondentID": 1003,
            "StartDate": "2026-02-01",
            "AgeGroup": "45-54",
            "Gender": "Male",
            "Home_parkrun": "Basingstoke",
            "How_often": "Once a month",
            "What_enjoy": "Keeping fit and tracking my times",
            "Improvements": "The results sometimes take a while",
            "Recommend": "Yes",
        },
    ]
    return make_csv_bytes(rows)


# ── DB mock ────────────────────────────────────────────────────────────────────


def _mock_execute(data: list | dict | None = None):
    """Return an object that looks like a supabase-py execute() result."""
    result = MagicMock()
    result.data = data if data is not None else []
    return result


class _MockQuery:
    """
    Self-chaining query object that returns a configurable result on execute().
    Mirrors the key Supabase PostgREST behaviours:
      .single() → execute() returns the first element (not the list), or None.
      All other chains return the full list.
    """

    def __init__(self, data: list | dict | None = None):
        self._data = data if data is not None else []
        self._single = False

    # Chainable stubs — every method returns self.
    def select(self, *a, **kw): return self
    def insert(self, *a, **kw): return self
    def update(self, *a, **kw): return self
    def upsert(self, *a, **kw): return self
    def delete(self, *a, **kw): return self
    def eq(self, *a, **kw): return self
    def neq(self, *a, **kw): return self
    def in_(self, *a, **kw): return self
    def is_(self, *a, **kw): return self
    def order(self, *a, **kw): return self
    def limit(self, *a, **kw): return self
    def lt(self, *a, **kw): return self

    def single(self, *a, **kw):
        self._single = True
        return self

    @property
    def not_(self): return self

    async def execute(self):
        if self._single:
            data = self._data[0] if self._data else None
        else:
            data = self._data
        return _mock_execute(data)


class MockDB:
    """
    Minimal Supabase client mock.

    Preset per-table data via the `tables` dict before the test runs:
      db.tables["ingestion_jobs"] = [{"id": "job-1", ...}]

    Any table not in the dict returns an empty list.
    """

    def __init__(self):
        self.tables: dict[str, list] = {}

    def table(self, name: str) -> _MockQuery:
        return _MockQuery(data=self.tables.get(name, []))

    def rpc(self, *a, **kw) -> _MockQuery:
        return _MockQuery(data=[])


def make_mock_db() -> MockDB:
    return MockDB()


@pytest.fixture
def mock_db():
    return make_mock_db()


@pytest.fixture
def client(mock_db):
    """FastAPI test client with the Supabase dependency overridden."""
    from app.database import get_db

    app.dependency_overrides[get_db] = lambda: mock_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def standard_csv():
    return make_standard_csv_bytes()


@pytest.fixture
def surveymonkey_csv():
    return make_surveymonkey_csv_bytes()
