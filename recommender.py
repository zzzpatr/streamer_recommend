"""Load, score, and rank streamer recommendations."""

from __future__ import annotations

import json
from typing import IO

import pandas as pd

from prepare_data import split_values
from search_index import TagIndex, build_inverted_index, score_preferences

FIELD_LABELS = {
    "gender": "性別",
    "personality": "性格定位",
    "appearance": "外型特徵",
    "talents": "才藝",
    "featured_topics": "直播主題",
    "live_streaming_style": "直播風格",
}
FIELD_WEIGHTS = {
    "gender": 0.25,
    "personality": 0.15,
    "appearance": 0.15,
    "talents": 0.15,
    "featured_topics": 0.15,
    "live_streaming_style": 0.15,
}
REASON_PREFIXES = {
    "gender": None,
    "personality": None,
    "appearance": "外型",
    "talents": "才藝",
    "featured_topics": "主題",
    "live_streaming_style": "風格",
}
REASON_VALUE_ALIASES = {
    ("gender", "男"): "男性",
    ("gender", "女"): "女性",
}
SEMANTIC_FALLBACK_THRESHOLD = 0.2


def build_ranking_query(
    preferences: dict[str, list[str]],
    semantic_query: str,
) -> str:
    """Build a complete query for semantic fallback and tag-score tie-breaking."""
    parts = [
        f"{FIELD_LABELS[field]}：{'、'.join(values)}"
        for field, values in preferences.items()
        if field in FIELD_LABELS and values
    ]
    if semantic_query.strip():
        parts.append(f"其他偏好：{semantic_query.strip()}")
    return "；".join(parts)


def parse_reasons(raw_reasons: object) -> dict[str, str]:
    """Parse a streamer's reason evidence from its JSON string."""
    if pd.isna(raw_reasons):
        return {}
    try:
        parsed = json.loads(str(raw_reasons))
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): str(value).strip()
        for key, value in parsed.items()
        if str(value).strip()
    }


def matched_reason_evidence(
    raw_reasons: object,
    matched_tags: list[str],
) -> list[tuple[str, str]]:
    """Return display labels and source evidence for matched preference tags."""
    reasons = parse_reasons(raw_reasons)
    evidence: list[tuple[str, str]] = []
    for matched_tag in matched_tags:
        if ":" not in matched_tag:
            continue
        field, value = matched_tag.split(":", 1)
        if field not in FIELD_LABELS:
            continue
        reason_value = REASON_VALUE_ALIASES.get((field, value), value)
        prefix = REASON_PREFIXES[field]
        expected_key = (
            reason_value
            if prefix is None
            else f"{prefix}_{reason_value}"
        )
        reason = reasons.get(expected_key) or reasons.get(reason_value)
        if reason:
            evidence.append((f"{FIELD_LABELS[field]}：{value}", reason))
    return evidence


def load_streamers(source: IO[bytes]) -> pd.DataFrame:
    """Load and validate streamer metadata from an uploaded CSV file."""
    streamers = pd.read_csv(source, dtype={"pfid": "string"})
    required_columns = {
        "pfid",
        *FIELD_LABELS,
        "overall_vibe",
        "reasons",
        "self_description",
    }
    missing_columns = sorted(required_columns.difference(streamers.columns))
    if missing_columns:
        raise ValueError("CSV 缺少必要欄位：" + "、".join(missing_columns))
    if streamers["pfid"].isna().any() or streamers["pfid"].duplicated().any():
        raise ValueError("pfid 不可為空且必須唯一")
    return streamers


def load_tag_index(streamers: pd.DataFrame) -> TagIndex:
    """Build an in-memory inverted index for the currently uploaded dataset."""
    rows = []
    for field in FIELD_LABELS:
        for pfid, raw_value in zip(
            streamers["pfid"], streamers[field], strict=True
        ):
            rows.extend(
                {"pfid": pfid, "field": field, "value": value}
                for value in split_values(raw_value)
            )
    tags = pd.DataFrame(rows, columns=["pfid", "field", "value"])
    return build_inverted_index(tags)


def _hard_filter_ids(
    tag_index: TagIndex,
    excluded_preferences: dict[str, list[str]],
) -> set[str]:
    excluded_ids: set[str] = set()
    for field, values in excluded_preferences.items():
        for value in values:
            excluded_ids.update(tag_index.get(field, {}).get(value, ()))
    return excluded_ids


def _apply_hard_filters(
    ranking: pd.DataFrame,
    tag_index: TagIndex,
    excluded_preferences: dict[str, list[str]],
) -> pd.DataFrame:
    excluded_ids = _hard_filter_ids(tag_index, excluded_preferences)
    if excluded_ids:
        ranking = ranking[~ranking["pfid"].isin(excluded_ids)]
    return ranking


def _apply_hidden_pfids(
    ranking: pd.DataFrame,
    hidden_pfids: set[str] | None,
) -> pd.DataFrame:
    if hidden_pfids:
        return ranking[~ranking["pfid"].isin(hidden_pfids)]
    return ranking


def _rank_exact_tags(
    tag_index: TagIndex,
    preferences: dict[str, list[str]],
    excluded_preferences: dict[str, list[str]],
) -> pd.DataFrame:
    ranking_preferences = {
        field: preferences.get(field, []) for field in FIELD_WEIGHTS
    }
    active_fields = [field for field, values in preferences.items() if values]
    active_fields = [field for field in active_fields if field in FIELD_WEIGHTS]
    active_total = sum(FIELD_WEIGHTS[field] for field in active_fields)
    active_weights = {
        field: FIELD_WEIGHTS[field] / active_total
        for field in active_fields
    } if active_total else {}

    tag_scores = score_preferences(
        tag_index, ranking_preferences, active_weights
    ).rename(
        columns={"match_score": "tag_score"}
    )
    if tag_scores.empty:
        return tag_scores
    tag_scores = _apply_hard_filters(
        tag_scores, tag_index, excluded_preferences
    )
    return tag_scores.sort_values(
        ["tag_score", "pfid"], ascending=[False, True]
    )


def rank_streamers(
    streamers: pd.DataFrame,
    tag_index: TagIndex,
    preferences: dict[str, list[str]],
    excluded_preferences: dict[str, list[str]],
    vector_scores: pd.DataFrame | None = None,
    literal_scores: pd.DataFrame | None = None,
    hidden_pfids: set[str] | None = None,
    top_n: int = 5,
    semantic_threshold: float = SEMANTIC_FALLBACK_THRESHOLD,
) -> pd.DataFrame:
    """Rank identity and explicit tags first, then use semantics."""
    exact = _rank_exact_tags(
        tag_index, preferences, excluded_preferences
    )
    has_vectors = vector_scores is not None and not vector_scores.empty
    has_literals = literal_scores is not None and not literal_scores.empty
    if not exact.empty or has_literals:
        if exact.empty:
            ranking = literal_scores.copy()
            ranking["tag_score"] = 0.0
            ranking["matched_tags"] = [[] for _ in range(len(ranking))]
        elif has_literals:
            ranking = exact.merge(literal_scores, on="pfid", how="outer")
            ranking["tag_score"] = ranking["tag_score"].fillna(0.0)
            ranking["matched_tags"] = ranking["matched_tags"].apply(
                lambda value: value if isinstance(value, list) else []
            )
        else:
            ranking = exact.copy()

        if "literal_score" not in ranking:
            ranking["literal_score"] = 0.0
            ranking["matched_literals"] = [[] for _ in range(len(ranking))]
        else:
            ranking["literal_score"] = ranking["literal_score"].fillna(0.0)
            ranking["matched_literals"] = ranking["matched_literals"].apply(
                lambda value: value if isinstance(value, list) else []
            )

        ranking = _apply_hard_filters(
            ranking, tag_index, excluded_preferences
        )
        ranking = _apply_hidden_pfids(ranking, hidden_pfids)
        ranking["retrieval_type"] = "exact"
        ranking["vector_score"] = 0.0
        ranking["match_score"] = ranking["tag_score"]
        if has_vectors:
            # Left join keeps the exact candidate set unchanged.
            ranking = ranking.drop(columns=["vector_score"]).merge(
                vector_scores, on="pfid", how="left"
            )
            ranking["vector_score"] = ranking["vector_score"].fillna(0.0)
            ranking["retrieval_type"] = "semantic_tiebreak"
        literal_mask = ranking["literal_score"] > 0
        ranking.loc[literal_mask, "retrieval_type"] = "literal_match"
        ranking.loc[literal_mask, "match_score"] = ranking.loc[
            literal_mask, "literal_score"
        ]
        ranking = ranking.sort_values(
            [
                "literal_score",
                "tag_score",
                "vector_score",
                "pfid",
            ],
            ascending=[False, False, False, True],
        ).head(top_n)
    elif has_vectors:
        ranking = vector_scores[
            vector_scores["vector_score"] >= semantic_threshold
        ].copy()
        ranking = _apply_hard_filters(
            ranking, tag_index, excluded_preferences
        )
        ranking = _apply_hidden_pfids(ranking, hidden_pfids)
        ranking["tag_score"] = 0.0
        ranking["matched_tags"] = [[] for _ in range(len(ranking))]
        ranking["literal_score"] = 0.0
        ranking["matched_literals"] = [[] for _ in range(len(ranking))]
        ranking["match_score"] = ranking["vector_score"]
        ranking["retrieval_type"] = "semantic_fallback"
        ranking = ranking.sort_values(
            ["vector_score", "pfid"],
            ascending=[False, True],
        ).head(top_n)
    else:
        return pd.DataFrame()

    if ranking.empty:
        return pd.DataFrame()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking.merge(
        streamers,
        on="pfid",
        how="left",
        validate="one_to_one",
    ).sort_values("rank")
