"""In-memory inverted index for exact streamer tag retrieval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd


TagIndex = dict[str, dict[str, frozenset[str]]]


def build_inverted_index(tags: pd.DataFrame) -> TagIndex:
    """Build field -> value -> pfid set from the normalized tag table."""
    required = {"pfid", "field", "value"}
    missing = required.difference(tags.columns)
    if missing:
        raise ValueError(f"Missing tag columns: {', '.join(sorted(missing))}")

    index: TagIndex = {}
    clean = tags.dropna(subset=["pfid", "field", "value"]).copy()
    clean["pfid"] = clean["pfid"].astype("string")

    for (field, value), group in clean.groupby(["field", "value"], sort=True):
        index.setdefault(str(field), {})[str(value)] = frozenset(group["pfid"])

    return index


def find_streamers(
    index: TagIndex,
    field: str,
    values: list[str],
    match: Literal["any", "all"] = "any",
) -> set[str]:
    """Return pfids matching any or all requested values in one field."""
    if not values:
        return set()

    matched_sets = [set(index.get(field, {}).get(value, ())) for value in values]
    if match == "any":
        return set().union(*matched_sets)
    if match == "all":
        return set.intersection(*matched_sets)
    raise ValueError("match must be 'any' or 'all'")


def score_preferences(
    index: TagIndex,
    preferences: dict[str, list[str]],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score pfids by weighted tag coverage and return highest scores first."""
    weights = weights or {}
    scores: Counter[str] = Counter()
    matched_tags: dict[str, list[str]] = {}

    active_fields = [field for field, values in preferences.items() if values]
    default_weight = 1 / len(active_fields) if active_fields else 0

    for field in active_fields:
        values = list(dict.fromkeys(preferences[field]))
        field_weight = weights.get(field, default_weight)
        score_per_tag = field_weight / len(values)

        for value in values:
            for pfid in index.get(field, {}).get(value, ()):
                scores[pfid] += score_per_tag
                matched_tags.setdefault(pfid, []).append(f"{field}:{value}")

    rows = [
        {
            "pfid": pfid,
            "match_score": round(score, 6),
            "matched_tags": matched_tags[pfid],
        }
        for pfid, score in scores.most_common()
    ]
    return pd.DataFrame(rows, columns=["pfid", "match_score", "matched_tags"])


def serialize_index(index: TagIndex) -> dict[str, dict[str, list[str]]]:
    """Convert frozensets to sorted lists for a deterministic JSON artifact."""
    return {
        field: {
            value: sorted(pfids)
            for value, pfids in sorted(value_index.items())
        }
        for field, value_index in sorted(index.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags", type=Path, default=Path("data_prepared/anchor_tags.csv")
    )
    parser.add_argument("--field", required=True)
    parser.add_argument("--value", action="append", required=True)
    parser.add_argument("--match", choices=["any", "all"], default="any")
    args = parser.parse_args()

    tags = pd.read_csv(args.tags, dtype={"pfid": "string"})
    index = build_inverted_index(tags)
    matched = sorted(find_streamers(index, args.field, args.value, args.match))
    print(json.dumps({"count": len(matched), "pfids": matched}, ensure_ascii=False))


if __name__ == "__main__":
    main()

