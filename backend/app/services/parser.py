"""
CSV / XLSX parser for parkrun survey exports.

Handles two formats:
  - Standard (Google Forms, manual): single header row
  - SurveyMonkey: two header rows (group name + question label)

SurveyMonkey two-row format example
────────────────────────────────────
Row 0: "Background"  "Background"  "Q1"                  "Q1"
Row 1: "Age group"   "Gender"      "Rate your experience" "Comments"
Row 2: "25-34"       "Female"      "5"                    "Great event!"

Row 0 is forward-filled so blank cells inherit the group name from the left.
The combined label becomes: "{group}: {question}" if the group is meaningful,
otherwise just "{question}".
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.models.surveys import ParsedSurvey


# Columns whose names suggest they are metadata, not survey questions.
_METADATA_PATTERNS = re.compile(
    r"(respondent.?id|collector.?id|start.?date|end.?date|ip.?address|email"
    r"|response.?id|timestamp|created.?at|modified.?at)",
    re.IGNORECASE,
)


def _is_metadata_column(name: str) -> bool:
    return bool(_METADATA_PATTERNS.search(name))


def _forward_fill(row: list[Any]) -> list[str]:
    """Forward-fill None/empty cells in a list, returning strings."""
    result: list[str] = []
    last = ""
    for cell in row:
        val = str(cell).strip() if cell is not None else ""
        if val:
            last = val
        result.append(last)
    return result


def _detect_format(df_raw: pd.DataFrame) -> Literal["surveymonkey", "standard"]:
    """
    Detect SurveyMonkey two-row header by checking if the first row has many
    duplicate values (group names repeat across sub-questions).
    A duplication ratio above 0.4 with at least 6 columns signals SurveyMonkey.
    """
    if len(df_raw) < 2 or len(df_raw.columns) < 6:
        return "standard"

    row0 = [str(v).strip() for v in df_raw.iloc[0] if str(v).strip()]
    if not row0:
        return "standard"

    unique_ratio = len(set(row0)) / len(row0)
    return "surveymonkey" if unique_ratio < 0.6 else "standard"


def _parse_standard(df: pd.DataFrame, file_name: str) -> ParsedSurvey:
    """Parse a CSV/XLSX with a single header row (Google Forms, manual exports)."""
    questions = []
    for pos, col in enumerate(df.columns):
        questions.append({"column_key": col, "label": col, "position": pos})

    rows = df.to_dict(orient="records")
    return ParsedSurvey(
        name=_name_from_filename(file_name),
        source="google_forms",
        questions=questions,
        rows=rows,
        row_count=len(rows),
        column_count=len(df.columns),
        file_name=file_name,
    )


def _parse_surveymonkey(df_raw: pd.DataFrame, file_name: str) -> ParsedSurvey:
    """
    Parse SurveyMonkey two-row header format.
    Row 0 = group/section names, Row 1 = question labels, Row 2+ = data.
    """
    groups = _forward_fill(list(df_raw.iloc[0]))
    questions_row = [str(v).strip() if v is not None else "" for v in df_raw.iloc[1]]

    labels: list[str] = []
    for group, question in zip(groups, questions_row):
        if group and group != question and not _is_metadata_column(group):
            labels.append(f"{group}: {question}" if question else group)
        else:
            labels.append(question or group)

    # Build a clean dataframe from row 2 onwards with proper column names.
    # Deduplicate identical labels by appending a suffix.
    unique_labels = _deduplicate(labels)
    data_df = df_raw.iloc[2:].copy()
    data_df.columns = unique_labels  # type: ignore[assignment]
    data_df = data_df.reset_index(drop=True)

    questions = [
        {"column_key": lbl, "label": lbl, "position": pos}
        for pos, lbl in enumerate(unique_labels)
    ]
    rows = data_df.to_dict(orient="records")

    return ParsedSurvey(
        name=_name_from_filename(file_name),
        source="surveymonkey",
        questions=questions,
        rows=rows,
        row_count=len(rows),
        column_count=len(unique_labels),
        file_name=file_name,
    )


def _deduplicate(labels: list[str]) -> list[str]:
    """Append _2, _3 … to duplicate column names."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            result.append(f"{label}_{seen[label]}")
        else:
            seen[label] = 1
            result.append(label)
    return result


def _name_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.replace("_", " ").replace("-", " ").title()


def _read_raw(content: bytes, file_name: str) -> pd.DataFrame:
    """Read raw bytes into a DataFrame with no header parsing (header=None)."""
    ext = Path(file_name).suffix.lower()
    buf = io.BytesIO(content)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(buf, header=None, dtype=str)
    return pd.read_csv(buf, header=None, dtype=str)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf"}


def _parse_pdf(content: bytes, file_name: str) -> ParsedSurvey:
    """
    Extract text from a PDF and map it to the survey data model.

    Each page becomes one response under a single 'Document Content' question.
    This lets the embedding, clustering, and wiki stages treat the document
    like any other open-ended survey — semantically searchable via chat.

    Raises ValueError if no text can be extracted (e.g. scanned/image PDFs).
    """
    try:
        from pypdf import PdfReader  # type: ignore[import]
    except ImportError as exc:
        raise ValueError(
            "PDF support requires pypdf. Run: pip install pypdf>=4.0.0"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Cannot read PDF '{file_name}': {exc}") from exc

    pages = [
        page.extract_text().strip()
        for page in reader.pages
        if page.extract_text() and page.extract_text().strip()
    ]

    if not pages:
        raise ValueError(
            f"No text could be extracted from '{file_name}'. "
            "The PDF may be scanned or image-based."
        )

    question = {"column_key": "content", "label": "Document Content", "position": 0}
    rows = [{"content": page} for page in pages]

    return ParsedSurvey(
        name=_name_from_filename(file_name),
        source="pdf",
        questions=[question],
        rows=rows,
        row_count=len(rows),
        column_count=1,
        file_name=file_name,
    )


def parse_survey_file(content: bytes, file_name: str) -> ParsedSurvey:
    """
    Main entry point. Accepts raw file bytes and returns a ParsedSurvey.
    Raises ValueError on malformed / unreadable files.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Cannot read file '{file_name}': unsupported extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return _parse_pdf(content, file_name)

    try:
        df_raw = _read_raw(content, file_name)
    except Exception as exc:
        raise ValueError(f"Cannot read file '{file_name}': {exc}") from exc

    if df_raw.empty or len(df_raw) < 2:
        raise ValueError(f"File '{file_name}' appears to have no data rows.")

    fmt = _detect_format(df_raw)

    if fmt == "surveymonkey":
        return _parse_surveymonkey(df_raw, file_name)

    buf = io.BytesIO(content)
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(buf, dtype=str)
    else:
        df = pd.read_csv(buf, dtype=str)

    return _parse_standard(df, file_name)


def infer_respondent_ref(row: dict[str, Any]) -> str | None:
    """Extract an anonymised respondent reference from a data row if present."""
    for key in row:
        if re.search(r"respondent.?id|response.?id|collector.?id", key, re.IGNORECASE):
            val = row[key]
            return str(val) if val and str(val) != "nan" else None
    return None


def infer_responded_at(row: dict[str, Any]) -> str | None:
    """Extract a response timestamp from a data row if present."""
    for key in row:
        if re.search(r"start.?date|end.?date|timestamp|responded.?at", key, re.IGNORECASE):
            val = row[key]
            return str(val) if val and str(val) != "nan" else None
    return None
