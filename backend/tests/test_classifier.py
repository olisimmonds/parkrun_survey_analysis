"""
Unit tests for the question type classifier.

External LLM calls are mocked so tests run offline without a Groq key.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.classifier import (
    VALID_TYPES,
    _parse_response,
    classify_questions,
    compute_distinct_values,
)


# ── _parse_response ───────────────────────────────────────────────────────────


def _q(label: str) -> dict:
    return {"column_key": label, "label": label, "position": 0}


def test_parse_response_valid_json():
    raw = json.dumps([
        {"column": "Age group", "type": "demographic"},
        {"column": "Rating", "type": "rating"},
        {"column": "Comments", "type": "open_ended"},
    ])
    result = _parse_response(raw, [_q("Age group"), _q("Rating"), _q("Comments")])
    assert result["Age group"] == "demographic"
    assert result["Rating"] == "rating"
    assert result["Comments"] == "open_ended"


def test_parse_response_strips_markdown_fences():
    raw = "```json\n[{\"column\": \"Gender\", \"type\": \"demographic\"}]\n```"
    result = _parse_response(raw, [_q("Gender")])
    assert result["Gender"] == "demographic"


def test_parse_response_invalid_type_falls_back():
    raw = json.dumps([{"column": "X", "type": "made_up_type"}])
    result = _parse_response(raw, [_q("X")])
    assert result["X"] == "open_ended"


def test_parse_response_missing_column_falls_back():
    raw = json.dumps([{"column": "A", "type": "rating"}])
    result = _parse_response(raw, [_q("A"), _q("B")])
    assert result["A"] == "rating"
    assert result["B"] == "open_ended"


def test_parse_response_bad_json_falls_back():
    result = _parse_response("this is not json at all", [_q("X"), _q("Y")])
    assert result["X"] == "open_ended"
    assert result["Y"] == "open_ended"


# ── compute_distinct_values ───────────────────────────────────────────────────


def test_compute_distinct_values_basic():
    rows = [
        {"Gender": "Male", "Rating": "5", "Comments": "Great"},
        {"Gender": "Female", "Rating": "4", "Comments": "Good event"},
        {"Gender": "Male", "Rating": "5", "Comments": "Loved it"},
    ]
    result = compute_distinct_values(rows)
    assert set(result["Gender"]) == {"Male", "Female"}
    assert set(result["Rating"]) == {"4", "5"}


def test_compute_distinct_values_ignores_nan():
    rows = [{"Col": "nan"}, {"Col": None}, {"Col": "Real value"}]
    result = compute_distinct_values(rows)
    assert result.get("Col") == ["Real value"]


def test_compute_distinct_values_truncates_at_max():
    rows = [{"Col": str(i)} for i in range(100)]
    result = compute_distinct_values(rows, max_per_col=10)
    assert len(result["Col"]) == 10


# ── classify_questions (mocked Groq) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_questions_returns_all_types():
    questions = [
        {"column_key": "age", "label": "Age group", "position": 0},
        {"column_key": "rating", "label": "Overall satisfaction (1-5)", "position": 1},
        {"column_key": "comments", "label": "What did you enjoy most?", "position": 2},
        {"column_key": "freq", "label": "How often do you attend?", "position": 3},
    ]

    mock_response_content = json.dumps([
        {"column": "Age group", "type": "demographic"},
        {"column": "Overall satisfaction (1-5)", "type": "rating"},
        {"column": "What did you enjoy most?", "type": "open_ended"},
        {"column": "How often do you attend?", "type": "multiple_choice"},
    ])

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = mock_response_content

    with patch("app.services.classifier.AsyncGroq") as MockGroq:
        mock_client = AsyncMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        result = await classify_questions(questions)

    assert len(result) == 4
    types = {q["label"]: q["question_type"] for q in result}
    assert types["Age group"] == "demographic"
    assert types["Overall satisfaction (1-5)"] == "rating"
    assert types["What did you enjoy most?"] == "open_ended"
    assert types["How often do you attend?"] == "multiple_choice"


@pytest.mark.asyncio
async def test_classify_questions_all_types_valid():
    questions = [{"column_key": f"col{i}", "label": f"Label {i}", "position": i} for i in range(5)]
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "invalid json !!!"

    with patch("app.services.classifier.AsyncGroq") as MockGroq:
        mock_client = AsyncMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        result = await classify_questions(questions)

    # All should fall back to open_ended when LLM returns garbage.
    for q in result:
        assert q["question_type"] in VALID_TYPES
        assert q["question_type"] == "open_ended"
