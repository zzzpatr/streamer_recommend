"""Pairwise preference-learning helpers for streamer exploration."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from prepare_data import split_values
from recommender import FIELD_LABELS


ProfileTags = dict[str, frozenset[str]]


def build_profile_tags(streamers: pd.DataFrame) -> ProfileTags:
    """Convert each streamer's structured fields into comparable field:value tags."""
    profiles: ProfileTags = {}
    for row in streamers.itertuples(index=False):
        tags = {
            f"{field}:{value}"
            for field in FIELD_LABELS
            for value in split_values(getattr(row, field))
        }
        profiles[str(row.pfid)] = frozenset(tags)
    return profiles


def jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    """Return 0 for identical profiles and 1 for disjoint profiles."""
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def update_weights(
    weights: dict[str, float],
    winner_tags: frozenset[str],
    loser_tags: frozenset[str],
    positive_step: float = 1.0,
    negative_step: float = 0.25,
) -> dict[str, float]:
    """Learn weak signals from pair-specific differences without hard exclusions."""
    updated = dict(weights)
    winner_only = winner_tags - loser_tags
    loser_only = loser_tags - winner_tags
    positive_per_tag = positive_step / len(winner_only) if winner_only else 0.0
    negative_per_tag = negative_step / len(loser_only) if loser_only else 0.0

    for tag in winner_only:
        updated[tag] = updated.get(tag, 0.0) + positive_per_tag
    for tag in loser_only:
        updated[tag] = updated.get(tag, 0.0) - negative_per_tag
    return updated


def rank_profiles(
    profiles: ProfileTags,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Rank every profile by accumulated pairwise tag signals."""
    rows = []
    for pfid, tags in profiles.items():
        positive = sorted(tag for tag in tags if weights.get(tag, 0.0) > 0)
        negative = sorted(tag for tag in tags if weights.get(tag, 0.0) < 0)
        rows.append(
            {
                "pfid": pfid,
                "pairwise_score": round(sum(weights.get(tag, 0.0) for tag in tags), 6),
                "positive_matches": positive,
                "negative_matches": negative,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pairwise_score", "pfid"],
        ascending=[False, True],
    )


def choose_pair(
    profiles: ProfileTags,
    weights: dict[str, float],
    seen_pfids: set[str],
    candidate_window: int = 20,
) -> tuple[str, str]:
    """Choose a contrasting initial pair, then contrast plausible top candidates."""
    available = sorted(set(profiles) - seen_pfids)
    if len(available) < 2:
        available = sorted(profiles)
    if len(available) < 2:
        raise ValueError("二選一至少需要兩位主播")

    if not weights:
        return max(
            combinations(available, 2),
            key=lambda pair: (
                jaccard_distance(profiles[pair[0]], profiles[pair[1]]),
                tuple(reversed(pair)),
            ),
        )

    ranked = rank_profiles(profiles, weights)
    ranked_available = [
        pfid for pfid in ranked["pfid"].tolist() if pfid in available
    ]
    first = ranked_available[0]
    alternatives = ranked_available[1 : candidate_window + 1]
    if not alternatives:
        alternatives = [pfid for pfid in available if pfid != first]
    second = max(
        alternatives,
        key=lambda pfid: (
            jaccard_distance(profiles[first], profiles[pfid]),
            -ranked_available.index(pfid),
        ),
    )
    return first, second


def strongest_signals(
    weights: dict[str, float],
    limit: int = 5,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return the strongest learned positive and negative tag signals."""
    positive = sorted(
        ((tag, weight) for tag, weight in weights.items() if weight > 0),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    negative = sorted(
        ((tag, weight) for tag, weight in weights.items() if weight < 0),
        key=lambda item: (item[1], item[0]),
    )[:limit]
    return positive, negative
