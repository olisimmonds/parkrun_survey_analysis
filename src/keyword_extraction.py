"""
Keyword Extraction
==================
Extracts meaningful keywords and phrases from free-text survey responses
using TF-IDF scoring. No external NLP downloads required.

Usage
-----
extractor = KeywordExtractor(custom_stopwords=["parkrun", "run"])
keywords = extractor.extract(responses, n=15)
# Returns: [("community", 0.45), ("volunteer", 0.38), ...]
"""

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# English stopwords (sklearn's built-in list — no download required)
# ---------------------------------------------------------------------------

ENGLISH_STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "just", "know", "let's", "like", "make",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "really", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "think", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "bit", "bit", "well", "could",
    "would", "really", "thing", "things", "bit", "bit", "lot", "lots",
    "always", "never", "every", "even", "still", "around", "without",
    "much", "many", "already", "back", "way", "feel", "felt", "find",
    "found", "take", "taken", "yes", "na", "n/a",
])


class KeywordExtractor:
    """
    Extracts keywords from a list of free-text responses using TF-IDF.

    Falls back to simple word frequency if there are too few responses
    to fit a TF-IDF model (fewer than 2 documents).
    """

    def __init__(self, custom_stopwords: List[str] = None):
        all_stops = set(ENGLISH_STOPWORDS)
        if custom_stopwords:
            all_stops.update(w.lower() for w in custom_stopwords)
        self.stopwords = all_stops

    def extract(self, responses: List[str], n: int = 15) -> List[Tuple[str, float]]:
        """
        Extract the top N keywords from a list of text responses.

        Parameters
        ----------
        responses : List of raw text strings.
        n         : Number of keywords to return.

        Returns
        -------
        List of (keyword, score) tuples, sorted by score descending.
        Score is a normalised TF-IDF weight (0–1) or a frequency ratio.
        """
        clean = [r for r in responses if isinstance(r, str) and len(r.strip()) > 5]
        if not clean:
            return []

        if len(clean) >= 2:
            return self._tfidf_keywords(clean, n)
        else:
            return self._frequency_keywords(clean, n)

    def format_for_prompt(self, keywords: List[Tuple[str, float]], n: int = 10) -> str:
        """
        Format top keywords as a readable string for LLM prompts.

        Example output: "community, volunteer, friendly, inclusive, route"
        """
        top = keywords[:n]
        return ", ".join(word for word, _ in top) if top else "No keywords extracted"

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _tfidf_keywords(self, texts: List[str], n: int) -> List[Tuple[str, float]]:
        """TF-IDF keyword extraction using scikit-learn."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(
                stop_words=list(self.stopwords),
                ngram_range=(1, 2),
                min_df=max(2, len(texts) // 20),  # appear in at least 5% of docs
                max_features=500,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # letters only, min 2 chars
            )
            matrix = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            # Mean TF-IDF score across all documents
            mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
            # Normalise to 0-1
            max_score = mean_scores.max()
            if max_score > 0:
                mean_scores = mean_scores / max_score
            # Sort descending
            top_indices = mean_scores.argsort()[::-1][:n]
            return [(feature_names[i], float(mean_scores[i])) for i in top_indices]
        except Exception:
            # Fall back to frequency if TF-IDF fails
            return self._frequency_keywords(texts, n)

    def _frequency_keywords(self, texts: List[str], n: int) -> List[Tuple[str, float]]:
        """Simple word-frequency fallback for small response sets."""
        import re
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            words = re.findall(r"[a-zA-Z]{3,}", text.lower())
            words = [w for w in words if w not in self.stopwords]
            counter.update(words)

        if not counter:
            return []

        total = sum(counter.values())
        top = counter.most_common(n)
        return [(word, count / total) for word, count in top]
