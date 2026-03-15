"""
Survey Parser
=============
Parses SurveyMonkey CSV and XLSX exports into structured data for analysis.

SurveyMonkey exports use a two-row header format:
  Row 0: Question stems (blank cells are forward-filled for multi-column questions)
  Row 1: Sub-question labels or answer option labels
  Row 2+: Response data (one row per respondent)

This module handles that format, strips metadata columns, detects question types,
and returns a clean SurveyData object ready for keyword extraction and LLM analysis.
"""

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class QuestionType(Enum):
    FREE_TEXT = "free_text"
    RATING = "rating"
    SINGLE_CHOICE = "single_choice"
    MULTI_SELECT = "multi_select"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        labels = {
            "free_text": "Open-Ended",
            "rating": "Rating / Scale",
            "single_choice": "Multiple Choice",
            "multi_select": "Select All That Apply",
            "unknown": "Unknown",
        }
        return labels.get(self.value, self.value)


@dataclass
class Question:
    """A single survey question and its parsed responses."""
    stem: str
    question_type: QuestionType
    sub_labels: List[str]

    # Type-specific response containers
    text_responses: List[str] = field(default_factory=list)    # FREE_TEXT
    numeric_values: List[float] = field(default_factory=list)  # RATING
    value_counts: Dict[str, int] = field(default_factory=dict) # CHOICE / MULTI_SELECT

    # Companion open-ended text for rating questions
    companion_texts: List[str] = field(default_factory=list)

    n_total: int = 0     # Total respondents
    n_answered: int = 0  # Non-empty responses


@dataclass
class SurveyData:
    """Fully parsed survey, ready for analysis."""
    source_name: str
    questions: List[Question]
    n_respondents: int
    metadata_df: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class SurveyParser:
    """
    Parses SurveyMonkey CSV and XLSX exports.

    Usage
    -----
    parser = SurveyParser(metadata_columns=["Respondent ID", ...])
    survey = parser.parse(file_bytes_or_path, filename="my_survey.csv")
    """

    # Labels in sub-header row that mark a companion open-ended column
    OPEN_ENDED_LABELS = {
        "open-ended response", "open ended response",
        "comment", "comments", "other", "please specify",
        "open-ended comment", "other (please specify)",
    }

    # Pattern for rating scale labels like "1 - Poor", "5 - Excellent"
    RATING_SCALE_PATTERN = re.compile(r"^\d+\s*[-–—]\s*.+")

    DEFAULT_METADATA_COLUMNS = [
        "Respondent ID", "Collector ID", "Start Date", "End Date",
        "IP Address", "Email Address", "First Name", "Last Name",
        "Custom Data 1", "Custom Data 2", "Custom Data 3",
    ]

    def __init__(self, metadata_columns: Optional[List[str]] = None):
        self.metadata_columns = set(
            c.lower() for c in (metadata_columns or self.DEFAULT_METADATA_COLUMNS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        source: Union[Path, bytes, io.BytesIO],
        filename: str = "survey.csv",
    ) -> SurveyData:
        """
        Parse a SurveyMonkey export file.

        Parameters
        ----------
        source   : File path, raw bytes, or BytesIO object.
        filename : Original filename — used to detect CSV vs XLSX format.

        Returns
        -------
        SurveyData with all detected questions parsed.
        """
        ext = Path(filename).suffix.lower()
        raw = self._read_raw(source, ext)

        if len(raw) < 3:
            raise ValueError(
                "The file has too few rows to be a valid SurveyMonkey export. "
                "Expected at least two header rows and one data row."
            )

        stems, sub_labels, data_df = self._split_headers(raw)
        meta_df, q_stems, q_subs, q_data = self._separate_metadata(
            stems, sub_labels, data_df
        )

        questions = self._build_questions(q_stems, q_subs, q_data)

        return SurveyData(
            source_name=filename,
            questions=questions,
            n_respondents=len(data_df),
            metadata_df=meta_df,
        )

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_raw(self, source: Union[Path, bytes, io.BytesIO], ext: str) -> pd.DataFrame:
        """Read file into a raw DataFrame with integer column indices and no header."""
        kwargs = dict(header=None, dtype=str, keep_default_na=False)

        if ext in (".csv", ".txt"):
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    if isinstance(source, (str, Path)):
                        return pd.read_csv(source, encoding=encoding, **kwargs)
                    else:
                        buf = io.BytesIO(source) if isinstance(source, bytes) else source
                        buf.seek(0)
                        return pd.read_csv(buf, encoding=encoding, **kwargs)
                except UnicodeDecodeError:
                    continue
            raise ValueError(
                "Could not decode the CSV file. Try re-saving it as UTF-8 from Excel."
            )

        if ext in (".xlsx", ".xls"):
            if isinstance(source, (str, Path)):
                return pd.read_excel(source, engine="openpyxl", **kwargs)
            else:
                buf = io.BytesIO(source) if isinstance(source, bytes) else source
                buf.seek(0)
                return pd.read_excel(buf, engine="openpyxl", **kwargs)

        raise ValueError(
            f"Unsupported file format: '{ext}'. Please upload a CSV or XLSX file."
        )

    # ------------------------------------------------------------------
    # Header detection and splitting
    # ------------------------------------------------------------------

    def _split_headers(
        self, raw: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Detect whether the file has a 1-row or 2-row header and split accordingly.

        Returns
        -------
        stems      : Forward-filled question stems for each column.
        sub_labels : Sub-question / answer-option labels for each column.
        data_df    : Response data with integer column indices.
        """
        row0 = raw.iloc[0].copy()
        row1 = raw.iloc[1].copy()

        # Heuristic: if the first cell of row1 is numeric, row1 is data (single header)
        first_cell = str(row1.iloc[0]).strip()
        is_double_header = not self._looks_numeric(first_cell)

        if is_double_header:
            stems = self._forward_fill(row0)
            sub_labels = row1.fillna("").astype(str).str.strip()
            data_df = raw.iloc[2:].reset_index(drop=True)
        else:
            stems = self._forward_fill(row0)
            sub_labels = pd.Series([""] * len(stems), dtype=str)
            data_df = raw.iloc[1:].reset_index(drop=True)

        data_df.columns = range(len(data_df.columns))
        return stems, sub_labels, data_df

    @staticmethod
    def _forward_fill(s: pd.Series) -> pd.Series:
        """Replace blank/nan cells with the last non-blank value (forward-fill)."""
        s = s.copy().astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return s.ffill().fillna("Unknown Question")

    # ------------------------------------------------------------------
    # Metadata separation
    # ------------------------------------------------------------------

    def _separate_metadata(
        self,
        stems: pd.Series,
        sub_labels: pd.Series,
        data_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
        """
        Split columns into metadata columns and question columns.

        Returns
        -------
        meta_df   : DataFrame of metadata columns (Respondent ID, dates, etc.)
        q_stems   : List of question stem strings.
        q_subs    : List of sub-label strings (one per column).
        q_data    : DataFrame of question response columns.
        """
        meta_idx, q_idx = [], []
        for i, stem in enumerate(stems):
            if str(stem).strip().lower() in self.metadata_columns:
                meta_idx.append(i)
            else:
                q_idx.append(i)

        if meta_idx:
            meta_df = data_df.iloc[:, meta_idx].copy()
            meta_df.columns = [str(stems.iloc[i]) for i in meta_idx]
        else:
            meta_df = pd.DataFrame()

        if q_idx:
            q_data = data_df.iloc[:, q_idx].copy()
            q_data.columns = range(len(q_idx))
            q_stems = [str(stems.iloc[i]) for i in q_idx]
            q_subs = [str(sub_labels.iloc[i]) for i in q_idx]
        else:
            q_data = data_df.copy()
            q_stems = list(stems.astype(str))
            q_subs = list(sub_labels.astype(str))

        return meta_df, q_stems, q_subs, q_data

    # ------------------------------------------------------------------
    # Question building
    # ------------------------------------------------------------------

    def _build_questions(
        self,
        stems: List[str],
        sub_labels: List[str],
        data_df: pd.DataFrame,
    ) -> List[Question]:
        """Group columns by stem and create Question objects."""
        # Group column indices by stem (preserving order)
        stem_to_indices: Dict[str, List[int]] = {}
        for i, stem in enumerate(stems):
            stem_to_indices.setdefault(stem, []).append(i)

        questions = []
        seen = set()
        for stem in stems:
            if stem in seen:
                continue
            seen.add(stem)
            indices = stem_to_indices[stem]
            subs = [sub_labels[i] for i in indices]
            cols = [data_df.iloc[:, i] for i in indices]
            questions.append(self._make_question(stem, subs, cols, len(data_df)))

        return questions

    def _make_question(
        self,
        stem: str,
        sub_labels: List[str],
        cols: List[pd.Series],
        n_total: int,
    ) -> Question:
        """Build a single Question by detecting its type and extracting responses."""
        q_type = self._detect_type(stem, sub_labels, cols)
        q = Question(
            stem=stem,
            question_type=q_type,
            sub_labels=sub_labels,
            n_total=n_total,
        )

        if q_type == QuestionType.FREE_TEXT:
            texts = self._clean_texts(cols[0])
            q.text_responses = texts
            q.n_answered = len(texts)

        elif q_type == QuestionType.RATING:
            for col, sub in zip(cols, sub_labels):
                if sub.strip().lower() in self.OPEN_ENDED_LABELS:
                    q.companion_texts = self._clean_texts(col)
                else:
                    nums = pd.to_numeric(col.replace("", np.nan), errors="coerce").dropna().tolist()
                    q.numeric_values.extend(nums)
            q.n_answered = len(q.numeric_values)

        elif q_type == QuestionType.MULTI_SELECT:
            for col, sub in zip(cols, sub_labels):
                label = sub.strip() or f"Option {len(q.value_counts) + 1}"
                selected = col.astype(str).str.strip()
                # SurveyMonkey marks selected options with the option text or "1"
                count = selected[
                    ~selected.isin(["", "nan", "0", "False", "false"])
                ].notna().sum()
                q.value_counts[label] = int(count)
            q.n_answered = sum(1 for v in q.value_counts.values() if v > 0)

        elif q_type == QuestionType.SINGLE_CHOICE:
            vals = self._clean_texts(cols[0])
            counts: Dict[str, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            q.value_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
            q.n_answered = len(vals)

        else:  # UNKNOWN — try to get text
            if cols:
                texts = self._clean_texts(cols[0])
                q.text_responses = texts
                q.n_answered = len(texts)

        return q

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def _detect_type(
        self,
        stem: str,
        sub_labels: List[str],
        cols: List[pd.Series],
    ) -> QuestionType:
        """Heuristically determine the question type from labels and data."""
        n_cols = len(cols)
        non_open_subs = [s for s in sub_labels if s.strip().lower() not in self.OPEN_ENDED_LABELS]
        actual_cols = [c for c, s in zip(cols, sub_labels) if s.strip().lower() not in self.OPEN_ENDED_LABELS]

        # Multiple non-companion columns → probably rating matrix or multi-select
        if len(actual_cols) > 1:
            # Check for rating scale pattern in sub-labels
            rating_matches = sum(
                1 for s in non_open_subs
                if self.RATING_SCALE_PATTERN.match(s.strip())
            )
            if rating_matches >= len(non_open_subs) // 2 and len(non_open_subs) > 1:
                return QuestionType.RATING

            # Check for binary/checkbox values (multi-select)
            first_col = actual_cols[0] if actual_cols else cols[0]
            unique_vals = set(first_col.dropna().astype(str).str.strip().unique())
            binary = {"0", "1", "", "x", "X", "true", "false", "True", "False", "checked", "nan"}
            if len(unique_vals - binary) <= 2:
                return QuestionType.MULTI_SELECT

            return QuestionType.MULTI_SELECT

        # Single column (or single non-companion column)
        primary_col = actual_cols[0] if actual_cols else cols[0]
        clean = primary_col.replace("", np.nan).dropna().astype(str).str.strip()
        clean = clean[clean.str.lower() != "nan"]

        if len(clean) == 0:
            return QuestionType.UNKNOWN

        # Check if values are all/mostly numeric
        numeric_vals = pd.to_numeric(clean, errors="coerce")
        numeric_ratio = numeric_vals.notna().sum() / len(clean)
        if numeric_ratio > 0.8:
            return QuestionType.RATING

        # Check uniqueness: low = categorical, high + long strings = free text
        unique_ratio = clean.nunique() / len(clean)
        avg_len = clean.str.len().mean()

        if unique_ratio < 0.20 and clean.nunique() <= 25:
            return QuestionType.SINGLE_CHOICE

        if avg_len > 25 or unique_ratio > 0.45:
            return QuestionType.FREE_TEXT

        return QuestionType.SINGLE_CHOICE

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_texts(col: pd.Series) -> List[str]:
        """Return non-empty string values from a Series."""
        texts = col.replace("", np.nan).dropna().astype(str).str.strip()
        return texts[texts.str.len() > 0].tolist()

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
