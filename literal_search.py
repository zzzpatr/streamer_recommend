"""Case-insensitive literal lookup over streamer metadata."""

from __future__ import annotations

import pandas as pd


LITERAL_SEARCH_FIELDS = (
    "pfid",
    "self_description",
)


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).casefold()


def search_literal_queries(
    streamers: pd.DataFrame,
    queries: list[str],
) -> pd.DataFrame:
    """Return streamers containing literal queries in searchable text fields."""
    query_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _normalize_text(query)
        if normalized and normalized not in seen:
            seen.add(normalized)
            query_pairs.append((" ".join(str(query).split()), normalized))
    if not query_pairs:
        return pd.DataFrame(
            columns=["pfid", "literal_score", "matched_literals"]
        )

    searchable_fields = [
        field for field in LITERAL_SEARCH_FIELDS if field in streamers.columns
    ]
    rows = []
    for _, row in streamers.iterrows():
        document = "\n".join(_normalize_text(row[field]) for field in searchable_fields)
        matched = [
            original
            for original, normalized in query_pairs
            if normalized in document
        ]
        if matched:
            rows.append(
                {
                    "pfid": str(row["pfid"]),
                    "literal_score": round(len(matched) / len(query_pairs), 6),
                    "matched_literals": matched,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["pfid", "literal_score", "matched_literals"],
    )
