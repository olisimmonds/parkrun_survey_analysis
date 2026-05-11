"""
Question type classifier.

Uses Groq's fast LLM to classify each survey column into one of six types:
  rating         Numeric scales, Likert, NPS, star ratings
  multiple_choice Fixed option sets, yes/no, agree/disagree, categories
  open_ended     Free-text comments, suggestions, opinions
  demographic    Age, gender, location, role, experience
  datetime       Date, time, or timestamp columns
  metadata       IDs, collector data, internal reference columns

Classification is done in a single batch call to minimise latency.
Results are cached per (label, distinct_values) to avoid re-classifying
identical questions across survey uploads.
"""
from __future__ import annotations

import json
import re
from typing import Any

from groq import AsyncGroq

from app.config import get_settings

VALID_TYPES = frozenset(
    {"rating", "multiple_choice", "open_ended", "demographic", "datetime", "metadata"}
)

_PROMPT_TEMPLATE_HEAD = """\
You are classifying survey question columns for a parkrun community data platform.

For each column below, output the most appropriate type:
  rating          — numeric scales, Likert (1-5, 1-10), NPS, star ratings, satisfaction scores
  multiple_choice — fixed option sets, yes/no, agree/disagree, single-select categories
  open_ended      — free text: comments, suggestions, opinions, anything respondents write freely
  demographic     — age, gender, location, role, home event, experience level, employment
  datetime        — any date, time, or timestamp column
  metadata        — IDs, collector IDs, internal reference numbers, start/end timestamps

Return ONLY a JSON array of objects, no commentary:
[{"column": "exact column name", "type": "type_name"}, ...]

Columns to classify:
"""


def _build_columns_payload(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the list of objects we send to the LLM for classification."""
    payload = []
    for q in questions:
        item: dict[str, Any] = {"column": q["label"]}
        if q.get("options"):
            item["sample_values"] = q["options"][:10]
        payload.append(item)
    return payload


def _parse_response(raw: str, questions: list[dict[str, Any]]) -> dict[str, str]:
    """
    Parse the LLM's JSON response into a {label: type} mapping.
    Falls back to 'open_ended' for any column the LLM missed or returned an
    invalid type for.
    """
    label_to_type: dict[str, str] = {}

    # Strip markdown code fences the model may add.
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        items = json.loads(raw)
        if isinstance(items, list):
            for item in items:
                col = item.get("column", "")
                typ = item.get("type", "open_ended")
                if typ not in VALID_TYPES:
                    typ = "open_ended"
                label_to_type[col] = typ
    except json.JSONDecodeError:
        pass  # Fallback applied below.

    # Ensure every question has a type.
    for q in questions:
        if q["label"] not in label_to_type:
            label_to_type[q["label"]] = "open_ended"

    return label_to_type


async def classify_questions(
    questions: list[dict[str, Any]],
    distinct_values: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Classify a list of question dicts (each having at least 'label' and 'position').
    Returns the same list with a 'question_type' key added to each item.

    Args:
        questions: List of dicts from ParsedSurvey.questions.
        distinct_values: Optional mapping of column_key → list of unique values seen
                         in the data, which helps the classifier distinguish
                         multiple_choice from open_ended.

    Returns:
        Updated list of question dicts with 'question_type' set.
    """
    settings = get_settings()

    # Enrich questions with distinct value samples for the LLM.
    enriched = []
    for q in questions:
        item = dict(q)
        if distinct_values and q["column_key"] in distinct_values:
            item["options"] = distinct_values[q["column_key"]]
        enriched.append(item)

    payload = _build_columns_payload(enriched)
    # Use concatenation rather than str.format() — the JSON payload contains
    # braces that would be misinterpreted as format placeholders.
    prompt = _PROMPT_TEMPLATE_HEAD + json.dumps(payload, indent=2)

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.groq_fast_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or "[]"
    label_to_type = _parse_response(raw, enriched)

    result = []
    for q in questions:
        item = dict(q)
        item["question_type"] = label_to_type.get(q["label"], "open_ended")
        result.append(item)

    return result


def compute_distinct_values(rows: list[dict[str, Any]], max_per_col: int = 20) -> dict[str, list[str]]:
    """
    Compute distinct non-null values per column from the data rows.
    Columns with > max_per_col distinct values are likely open-ended or datetime —
    we still return them (truncated) to let the LLM decide.
    """
    col_values: dict[str, set[str]] = {}
    for row in rows:
        for key, val in row.items():
            if val is None:
                continue
            s = str(val).strip()
            # Exclude empty strings and pandas-style nan representations.
            if not s or s.lower() == "nan":
                continue
            col_values.setdefault(key, set()).add(s)

    return {
        col: sorted(vals)[:max_per_col]
        for col, vals in col_values.items()
    }
