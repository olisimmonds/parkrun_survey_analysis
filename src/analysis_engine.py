"""
Analysis Engine
===============
Orchestrates the full analysis pipeline:

  1. Basic statistics (no LLM required)
  2. Keyword extraction
  3. LLM-powered narrative analysis (streamed)
  4. Chat context retrieval (TF-IDF based)
  5. Chat response generation (streamed)

Usage
-----
engine = AnalysisEngine(survey_data, settings, prompts, provider)

for question in survey_data.questions:
    stats = engine.get_basic_stats(question)
    keywords = engine.get_keywords(question)
    for chunk in engine.stream_llm_analysis(question, stats, keywords):
        print(chunk, end="")

# After analysis is complete, build chat context
engine.build_chat_context(all_analyses)

# Answer user chat questions
for chunk in engine.answer_chat("What are the main themes?", history):
    print(chunk, end="")
"""

import statistics
from typing import Dict, Generator, List, Optional, Tuple

from .keyword_extraction import KeywordExtractor
from .llm_interface import LLMProvider
from .survey_parser import Question, QuestionType, SurveyData


# ---------------------------------------------------------------------------
# Analysis Engine
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """
    Coordinates survey analysis: statistics, keywords, LLM insights, and chat.
    """

    def __init__(
        self,
        survey_data: SurveyData,
        settings: dict,
        prompts: dict,
        provider: Optional[LLMProvider] = None,
    ):
        self.survey_data = survey_data
        self.settings = settings
        self.prompts = prompts
        self.provider = provider

        custom_stops = settings.get("parser", {}).get("custom_stopwords", [])
        self.extractor = KeywordExtractor(custom_stopwords=custom_stops)
        self._context_retriever: Optional["ContextRetriever"] = None

    # ------------------------------------------------------------------
    # Basic statistics (no LLM)
    # ------------------------------------------------------------------

    def get_basic_stats(self, question: Question) -> Dict:
        """
        Compute non-LLM statistics for a question.

        Returns a dict with keys depending on question type:
          FREE_TEXT   : count, answered_pct, avg_words, total_words
          RATING      : count, mean, median, std, min, max, distribution
          SINGLE_CHOICE / MULTI_SELECT : count, value_counts, top_choice
        """
        q_type = question.question_type
        n_total = question.n_total
        n_answered = question.n_answered

        base = {
            "n_total": n_total,
            "n_answered": n_answered,
            "answered_pct": round(n_answered / n_total * 100, 1) if n_total > 0 else 0,
        }

        if q_type == QuestionType.FREE_TEXT:
            if question.text_responses:
                word_counts = [len(r.split()) for r in question.text_responses]
                base.update({
                    "avg_words": round(statistics.mean(word_counts), 1),
                    "median_words": statistics.median(word_counts),
                    "total_words": sum(word_counts),
                })
            else:
                base.update({"avg_words": 0, "median_words": 0, "total_words": 0})

        elif q_type == QuestionType.RATING:
            vals = question.numeric_values
            if vals:
                base.update({
                    "mean": round(statistics.mean(vals), 2),
                    "median": statistics.median(vals),
                    "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
                    "min": min(vals),
                    "max": max(vals),
                    "distribution": self._build_distribution(vals),
                })
            else:
                base.update({"mean": None, "median": None, "std": None,
                             "min": None, "max": None, "distribution": {}})

        elif q_type in (QuestionType.SINGLE_CHOICE, QuestionType.MULTI_SELECT):
            vc = question.value_counts
            total_selections = sum(vc.values()) if vc else 0
            pct = {
                k: round(v / total_selections * 100, 1)
                for k, v in vc.items()
                if total_selections > 0
            }
            base.update({
                "value_counts": vc,
                "value_pcts": pct,
                "top_choice": max(vc, key=vc.get) if vc else None,
            })

        return base

    @staticmethod
    def _build_distribution(values: List[float]) -> Dict[str, int]:
        """Build a frequency distribution for numeric values."""
        dist: Dict[str, int] = {}
        for v in values:
            key = str(int(round(v)))
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: float(x[0])))

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def get_keywords(self, question: Question) -> List[Tuple[str, float]]:
        """Extract top keywords from a question's text responses."""
        n = self.settings.get("analysis", {}).get("keywords_count", 15)
        texts = question.text_responses + question.companion_texts
        return self.extractor.extract(texts, n=n)

    # ------------------------------------------------------------------
    # LLM analysis (streaming)
    # ------------------------------------------------------------------

    def stream_llm_analysis(
        self,
        question: Question,
        stats: Dict,
        keywords: List[Tuple[str, float]],
    ) -> Generator[str, None, None]:
        """
        Stream an LLM analysis of a single question.

        Yields string chunks of the response.
        Yields an empty generator if no provider is configured.
        """
        if self.provider is None:
            return

        prompt = self._build_analysis_prompt(question, stats, keywords)
        system = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        yield from self.provider.stream_chat(messages)

    def _build_system_prompt(self) -> str:
        parkrun_info = self.prompts.get("parkrun_info", "")
        template = self.prompts.get("system_prompt", "You are a helpful analyst.")
        return template.format(parkrun_info=parkrun_info)

    def _build_analysis_prompt(
        self,
        question: Question,
        stats: Dict,
        keywords: List[Tuple[str, float]],
    ) -> str:
        """Build the user-facing analysis prompt for a question."""
        q_type = question.question_type
        parkrun_info = self.prompts.get("parkrun_info", "")
        max_samples = self.settings.get("analysis", {}).get("max_verbatim_samples", 5)

        if q_type == QuestionType.FREE_TEXT:
            template = self.prompts.get("free_text_analysis", "Analyse: {question}\n{stats}\n{keywords}\n{responses}")
            kw_str = self.extractor.format_for_prompt(keywords)
            samples = question.text_responses[:max_samples]
            responses_str = "\n".join(f'- "{r}"' for r in samples) if samples else "(No responses)"
            stats_str = (
                f"{stats.get('n_answered', 0)} responses "
                f"({stats.get('answered_pct', 0)}% response rate), "
                f"average {stats.get('avg_words', 0)} words per response"
            )
            return template.format(
                question=question.stem,
                stats=stats_str,
                keywords=kw_str,
                responses=responses_str,
                parkrun_info=parkrun_info,
            )

        elif q_type == QuestionType.RATING:
            template = self.prompts.get("rating_analysis", "Analyse rating: {question}\n{stats}")
            stats_str = (
                f"Mean: {stats.get('mean')}, "
                f"Median: {stats.get('median')}, "
                f"Std dev: {stats.get('std')}, "
                f"Range: {stats.get('min')}–{stats.get('max')}, "
                f"n={stats.get('n_answered')} responses ({stats.get('answered_pct')}% response rate)"
            )
            dist = stats.get("distribution", {})
            dist_str = ", ".join(f"{k}: {v}" for k, v in dist.items())

            companion_str = ""
            if question.companion_texts:
                samples = question.companion_texts[:max_samples]
                companion_str = "\nOpen-ended comments:\n" + "\n".join(f'- "{r}"' for r in samples)

            return template.format(
                question=question.stem,
                stats=stats_str + (f"\nDistribution: {dist_str}" if dist_str else ""),
                responses=companion_str,
                parkrun_info=parkrun_info,
            )

        elif q_type == QuestionType.MULTI_SELECT:
            template = self.prompts.get("multi_select_analysis", "Analyse: {question}\n{stats}")
            total = sum(question.value_counts.values()) or 1
            n_respondents = stats.get("n_total", 1) or 1
            stats_str = "\n".join(
                f"  {k}: {v} selections ({round(v / n_respondents * 100, 1)}% of respondents)"
                for k, v in sorted(question.value_counts.items(), key=lambda x: -x[1])
            )
            return template.format(
                question=question.stem,
                stats=stats_str,
                parkrun_info=parkrun_info,
            )

        else:  # SINGLE_CHOICE or UNKNOWN
            template = self.prompts.get("choice_analysis", "Analyse: {question}\n{stats}")
            total = sum(question.value_counts.values()) or 1
            stats_str = "\n".join(
                f"  {k}: {v} ({round(v / total * 100, 1)}%)"
                for k, v in sorted(question.value_counts.items(), key=lambda x: -x[1])[:15]
            )
            return template.format(
                question=question.stem,
                stats=stats_str or "(No data)",
                parkrun_info=parkrun_info,
            )

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def build_chat_context(self, analyses: List[Dict]) -> None:
        """
        Fit the TF-IDF context retriever on completed question analyses.

        Call this after all LLM analyses are complete. The retriever is
        then used to find the most relevant analysis for each chat query.

        Parameters
        ----------
        analyses : List of dicts, each with keys:
                   "stem" (str), "llm_text" (str), "keywords" (list)
        """
        self._context_retriever = ContextRetriever(analyses)
        self._context_retriever.fit()

    def answer_chat(
        self,
        user_message: str,
        history: List[dict],
    ) -> Generator[str, None, None]:
        """
        Stream a chat response to a user question about the survey.

        Uses TF-IDF retrieval to find the most relevant analysis context
        before sending the query to the LLM.
        """
        if self.provider is None:
            yield "No LLM provider configured. Please add your API key in the Settings page."
            return

        # Retrieve relevant context
        if self._context_retriever is not None:
            max_chars = self.settings.get("analysis", {}).get("chat_context_max_chars", 3000)
            top_k = self.settings.get("analysis", {}).get("chat_context_top_k", 3)
            context = self._context_retriever.retrieve(user_message, top_k=top_k, max_chars=max_chars)
        else:
            context = "No analysis has been run yet. Please run analysis first."

        parkrun_info = self.prompts.get("parkrun_info", "")
        chat_template = self.prompts.get("chat_system", "{context}")
        system_prompt = chat_template.format(
            parkrun_info=parkrun_info,
            context=context,
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        yield from self.provider.stream_chat(messages)


# ---------------------------------------------------------------------------
# TF-IDF Context Retriever for chat
# ---------------------------------------------------------------------------

class ContextRetriever:
    """
    Retrieves the most relevant question analyses for a chat query
    using TF-IDF cosine similarity.

    This is a lightweight alternative to vector embeddings — no external
    API calls or large model downloads required.
    """

    def __init__(self, analyses: List[Dict]):
        """
        Parameters
        ----------
        analyses : List of dicts with "stem", "llm_text", and "keywords" keys.
        """
        self.analyses = analyses
        self._vectorizer = None
        self._matrix = None
        self._documents: List[str] = []

    def fit(self) -> None:
        """Build the TF-IDF index over all question analyses."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._documents = []
            for a in self.analyses:
                stem = a.get("stem", "")
                llm_text = a.get("llm_text", "")
                kw = a.get("keywords", [])
                kw_str = " ".join(w for w, _ in kw) if kw else ""
                doc = f"{stem} {kw_str} {llm_text}"
                self._documents.append(doc)

            if not self._documents:
                return

            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
            self._matrix = self._vectorizer.fit_transform(self._documents)
        except Exception:
            # If sklearn fails, fall back to no retrieval
            self._vectorizer = None

    def retrieve(self, query: str, top_k: int = 3, max_chars: int = 3000) -> str:
        """
        Find the most relevant analyses for a given query.

        Returns
        -------
        Formatted string of the top-K analyses concatenated together.
        """
        if self._vectorizer is None or self._matrix is None:
            # No index — return all analyses truncated
            return self._format_all(max_chars)

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix)[0]
            top_indices = np.argsort(scores)[::-1][:top_k]

            parts = []
            total = 0
            for i in top_indices:
                a = self.analyses[i]
                text = f"Question: {a.get('stem', '')}\n{a.get('llm_text', '')}"
                if total + len(text) > max_chars:
                    text = text[: max_chars - total]
                    parts.append(text)
                    break
                parts.append(text)
                total += len(text)

            return "\n\n---\n\n".join(parts) if parts else "No relevant analysis found."
        except Exception:
            return self._format_all(max_chars)

    def _format_all(self, max_chars: int) -> str:
        parts = []
        total = 0
        for a in self.analyses:
            text = f"Question: {a.get('stem', '')}\n{a.get('llm_text', '')}"
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts) if parts else "No analysis available."
