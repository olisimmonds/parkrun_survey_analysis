"""
HDBSCAN clustering service for open-ended survey answers.

Clusters the 768-dim embeddings of open-ended answers per question,
then calls the LLM (Groq) to:
  1. Generate a concise theme label for each cluster.
  2. Write a 2–3 sentence summary describing the cluster.

Noise points (HDBSCAN label -1) are not stored as clusters.

Minimum cluster size adapts to the number of responses:
  < 50  responses → min_cluster_size = 3
  50–200          → min_cluster_size = 5
  200–1000        → min_cluster_size = 10
  > 1000          → min_cluster_size = 15
"""
from __future__ import annotations

import json
import logging
import re
from uuid import UUID

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)


def _adaptive_min_cluster_size(n: int) -> int:
    if n < 50:
        return 3
    if n < 200:
        return 5
    if n < 1000:
        return 10
    return 15


def cluster_embeddings(
    embeddings: list[list[float]],
    answer_ids: list[str],
    texts: list[str],
) -> list[dict]:
    """
    Run HDBSCAN on a set of embeddings.

    Returns a list of cluster dicts:
    {
      "cluster_id": int,
      "member_ids": [answer_id, ...],
      "centroid": [float, ...],
      "representative_ids": [answer_id, ...],   # 3 closest to centroid
      "representative_texts": [str, ...],
    }
    Noise points are excluded.
    """
    import hdbscan  # type: ignore

    if len(embeddings) < 3:
        log.warning("Too few embeddings (%d) to cluster — skipping.", len(embeddings))
        return []

    arr = np.array(embeddings, dtype=np.float32)
    min_size = _adaptive_min_cluster_size(len(arr))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels: np.ndarray = clusterer.fit_predict(arr)

    unique_labels = sorted(set(labels.tolist()) - {-1})
    if not unique_labels:
        log.info("HDBSCAN produced no clusters (all noise).")
        return []

    clusters = []
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        member_indices = np.where(mask)[0].tolist()
        member_embeddings = arr[mask]
        centroid = member_embeddings.mean(axis=0)

        # Find 3 answers closest to centroid.
        dists = np.linalg.norm(member_embeddings - centroid, axis=1)
        closest = np.argsort(dists)[:3].tolist()
        rep_indices = [member_indices[i] for i in closest]

        clusters.append({
            "cluster_id": cluster_id,
            "member_ids": [answer_ids[i] for i in member_indices],
            "centroid": centroid.tolist(),
            "representative_ids": [answer_ids[i] for i in rep_indices],
            "representative_texts": [texts[i] for i in rep_indices],
        })

    log.info(
        "Clustered %d responses into %d clusters (%d noise).",
        len(embeddings), len(clusters), int((labels == -1).sum()),
    )
    return clusters


_LABEL_PROMPT = """\
Below are representative verbatim quotes from a cluster of survey responses.
Respondents are parkrun community members (participants, volunteers, or Run Directors).

Quotes:
{quotes}

Produce a JSON object with two keys:
  "label"   — a concise 3-6 word theme label (title case, no quotes)
  "summary" — a 2-3 sentence description of the cluster's main finding

Return ONLY the JSON object, no commentary."""


async def label_clusters(
    clusters: list[dict],
    question_label: str,
) -> list[dict]:
    """
    Call Groq to add 'label' and 'summary' to each cluster dict.
    Processes clusters one at a time to keep prompts focused.
    Returns the same list with 'label' and 'summary' filled in.
    """
    from groq import AsyncGroq  # type: ignore

    settings = get_settings()
    client = AsyncGroq(api_key=settings.groq_api_key)

    labeled = []
    for cluster in clusters:
        quotes_block = "\n".join(
            f'- "{t}"' for t in cluster["representative_texts"]
        )
        prompt = _LABEL_PROMPT.format(quotes=quotes_block)

        try:
            resp = await client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or "{}"
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            parsed = json.loads(raw)
            label = str(parsed.get("label", f"Theme {cluster['cluster_id'] + 1}"))
            summary = str(parsed.get("summary", ""))
        except Exception as exc:
            log.warning("Cluster labelling failed for cluster %d: %s", cluster["cluster_id"], exc)
            label = f"Theme {cluster['cluster_id'] + 1}"
            summary = ""

        labeled.append({**cluster, "label": label, "summary": summary})

    return labeled
