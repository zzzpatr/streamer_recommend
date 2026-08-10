"""Direct controls for replacing recommendation results and finding similars."""

from __future__ import annotations

import numpy as np
import pandas as pd


EmbeddingIndex = tuple[np.ndarray, np.ndarray, dict]


def streamer_similarity_scores(
    embedding_index: EmbeddingIndex,
    anchor_pfid: str,
) -> pd.DataFrame:
    """Compare all cached streamer vectors with one selected streamer."""
    pfids, embeddings, _ = embedding_index
    anchor_matches = np.flatnonzero(pfids.astype(str) == str(anchor_pfid))
    if len(anchor_matches) != 1:
        raise ValueError(f"找不到唯一的相似主播基準：{anchor_pfid}")
    anchor_vector = embeddings[int(anchor_matches[0])]
    scores = np.clip(embeddings @ anchor_vector, 0.0, 1.0)
    return pd.DataFrame(
        {
            "pfid": pfids.astype(str),
            "vector_score": scores,
        }
    )


def dismiss_current_batch(
    dismissed_pfids: set[str],
    current_pfids: list[str],
) -> set[str]:
    """Return a new set that skips the currently displayed recommendation batch."""
    return set(dismissed_pfids) | {str(pfid) for pfid in current_pfids}
