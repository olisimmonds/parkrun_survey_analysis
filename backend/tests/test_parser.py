"""
Unit tests for the CSV/XLSX parser.

Tests cover:
  - Standard single-header CSV (Google Forms style)
  - SurveyMonkey two-row header CSV
  - Edge cases: empty files, bad extensions, duplicate column names
  - respondent_ref and responded_at extraction helpers
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from app.services.parser import (
    _detect_format,
    _forward_fill,
    infer_respondent_ref,
    infer_responded_at,
    parse_survey_file,
)
from tests.conftest import make_standard_csv_bytes, make_surveymonkey_csv_bytes


# ── _forward_fill ─────────────────────────────────────────────────────────────


def test_forward_fill_basic():
    row = ["Background", None, None, "Experience", None, "Other"]
    result = _forward_fill(row)
    assert result == ["Background", "Background", "Background", "Experience", "Experience", "Other"]


def test_forward_fill_all_filled():
    row = ["A", "B", "C"]
    assert _forward_fill(row) == ["A", "B", "C"]


def test_forward_fill_empty_start():
    row = [None, None, "Value"]
    result = _forward_fill(row)
    # Leading empty cells produce empty strings until first non-empty.
    assert result[0] == ""
    assert result[2] == "Value"


# ── _detect_format ────────────────────────────────────────────────────────────


def test_detect_format_surveymonkey():
    lines = (
        "Background,Background,Background,Experience,Experience,Experience\n"
        "Age group,Gender,Home parkrun,Satisfaction,Enjoy,Suggestions\n"
        "25-34,F,Southampton,5,Community,More parking\n"
    )
    df = pd.read_csv(io.StringIO(lines), header=None, dtype=str)
    fmt = _detect_format(df)
    assert fmt == "surveymonkey"


def test_detect_format_standard():
    lines = "RespondentID,AgeGroup,Gender,Satisfaction,Comments\n1001,25-34,M,5,Great\n"
    df = pd.read_csv(io.StringIO(lines), header=None, dtype=str)
    fmt = _detect_format(df)
    assert fmt == "standard"


def test_detect_format_too_few_columns():
    lines = "A,B\n1,2\n"
    df = pd.read_csv(io.StringIO(lines), header=None, dtype=str)
    # Fewer than 6 columns → always standard.
    assert _detect_format(df) == "standard"


# ── parse_survey_file: standard CSV ──────────────────────────────────────────


def test_parse_standard_csv_row_count(standard_csv):
    result = parse_survey_file(standard_csv, "participant_survey.csv")
    assert result.row_count == 3


def test_parse_standard_csv_column_count(standard_csv):
    result = parse_survey_file(standard_csv, "participant_survey.csv")
    assert result.column_count == 9


def test_parse_standard_csv_question_labels(standard_csv):
    result = parse_survey_file(standard_csv, "participant_survey.csv")
    labels = [q["label"] for q in result.questions]
    assert "What_enjoy" in labels
    assert "Improvements" in labels


def test_parse_standard_csv_source(standard_csv):
    result = parse_survey_file(standard_csv, "my_form.csv")
    assert result.source == "google_forms"


def test_parse_standard_csv_name_derived_from_filename(standard_csv):
    result = parse_survey_file(standard_csv, "volunteer_feedback_2024.csv")
    assert result.name == "Volunteer Feedback 2024"


def test_parse_standard_csv_rows_are_dicts(standard_csv):
    result = parse_survey_file(standard_csv, "x.csv")
    assert isinstance(result.rows[0], dict)
    assert "AgeGroup" in result.rows[0]


# ── parse_survey_file: SurveyMonkey CSV ──────────────────────────────────────


def test_parse_surveymonkey_detects_format(surveymonkey_csv):
    result = parse_survey_file(surveymonkey_csv, "survey.csv")
    assert result.source == "surveymonkey"


def test_parse_surveymonkey_row_count(surveymonkey_csv):
    result = parse_survey_file(surveymonkey_csv, "survey.csv")
    assert result.row_count == 5


def test_parse_surveymonkey_column_count(surveymonkey_csv):
    result = parse_survey_file(surveymonkey_csv, "survey.csv")
    assert result.column_count == 6


def test_parse_surveymonkey_label_combines_group_and_question(surveymonkey_csv):
    result = parse_survey_file(surveymonkey_csv, "survey.csv")
    labels = [q["label"] for q in result.questions]
    # Labels should combine group + question text.
    assert any("Age group" in lbl for lbl in labels)
    assert any("enjoy" in lbl.lower() for lbl in labels)


def test_parse_surveymonkey_data_rows_not_header(surveymonkey_csv):
    result = parse_survey_file(surveymonkey_csv, "survey.csv")
    # First data row should not contain "Background" (the group header).
    for row in result.rows:
        for val in row.values():
            assert val != "Background", "Header row leaked into data"


# ── parse_survey_file: XLSX ───────────────────────────────────────────────────


def test_parse_xlsx(tmp_path):
    df = pd.DataFrame({
        "AgeGroup": ["25-34", "35-44"],
        "Satisfaction": [5, 4],
        "Comments": ["Great event", "Loved the atmosphere"],
    })
    path = tmp_path / "survey.xlsx"
    df.to_excel(path, index=False)
    with open(path, "rb") as f:
        content = f.read()
    result = parse_survey_file(content, "survey.xlsx")
    assert result.row_count == 2
    assert result.column_count == 3


# ── parse_survey_file: error cases ───────────────────────────────────────────


def test_parse_bad_extension_raises():
    with pytest.raises(ValueError, match="Cannot read"):
        parse_survey_file(b"not a real file", "survey.json")


def test_parse_empty_csv_raises():
    with pytest.raises(ValueError, match="no data rows"):
        parse_survey_file(b"col1,col2\n", "empty.csv")


def test_parse_single_header_row_csv():
    """A CSV with only a header and no data rows should raise."""
    with pytest.raises(ValueError):
        parse_survey_file(b"A,B,C\n", "header_only.csv")


# ── infer_respondent_ref ──────────────────────────────────────────────────────


def test_infer_respondent_ref_found():
    row = {"RespondentID": "1001", "AgeGroup": "25-34"}
    assert infer_respondent_ref(row) == "1001"


def test_infer_respondent_ref_none_when_absent():
    row = {"AgeGroup": "25-34", "Comments": "Great"}
    assert infer_respondent_ref(row) is None


def test_infer_respondent_ref_ignores_nan():
    row = {"Respondent_ID": "nan"}
    assert infer_respondent_ref(row) is None


# ── infer_responded_at ────────────────────────────────────────────────────────


def test_infer_responded_at_start_date():
    row = {"StartDate": "2026-02-01 09:00", "AgeGroup": "25-34"}
    assert infer_responded_at(row) == "2026-02-01 09:00"


def test_infer_responded_at_none_when_absent():
    row = {"AgeGroup": "25-34"}
    assert infer_responded_at(row) is None


# ── PDF parsing ───────────────────────────────────────────────────────────────


def test_pdf_not_in_old_extension_guard():
    """PDF files no longer rejected by the extension check."""
    from app.services.parser import SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_pdf_with_no_text_raises():
    """An unreadable PDF (scanned / corrupted) raises a clear ValueError."""
    # An empty PDF stream that pypdf can open but has no extractable text.
    # We mock PdfReader.pages to return a page with no text.
    import unittest.mock as mock
    from app.services.parser import _parse_pdf

    fake_page = mock.MagicMock()
    fake_page.extract_text.return_value = ""
    fake_reader = mock.MagicMock()
    fake_reader.pages = [fake_page]

    with mock.patch("pypdf.PdfReader", return_value=fake_reader):
        with pytest.raises(ValueError, match="No text could be extracted"):
            _parse_pdf(b"%PDF-1.4", "empty.pdf")


def test_pdf_pages_become_rows():
    """Each page with text becomes a separate row in the parsed survey."""
    import unittest.mock as mock
    from app.services.parser import _parse_pdf

    pages_text = ["Page one content", "Page two content", "   ", "Page four"]
    fake_pages = []
    for t in pages_text:
        p = mock.MagicMock()
        p.extract_text.return_value = t
        fake_pages.append(p)

    fake_reader = mock.MagicMock()
    fake_reader.pages = fake_pages

    with mock.patch("pypdf.PdfReader", return_value=fake_reader):
        result = _parse_pdf(b"%PDF-1.4", "my_document.pdf")

    # Blank/whitespace pages are skipped
    assert result.row_count == 3
    assert result.column_count == 1
    assert result.source == "pdf"
    assert result.questions[0]["label"] == "Document Content"
    assert result.rows[0]["content"] == "Page one content"
    assert result.rows[2]["content"] == "Page four"
