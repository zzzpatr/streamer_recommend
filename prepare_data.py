"""Normalize and explode the streamer metadata into search-friendly tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from search_index import build_inverted_index, serialize_index


MULTI_VALUE_COLUMNS = [
    "personality",
    "appearance",
    "talents",
    "featured_topics",
    "live_streaming_style",
]
INDEXED_COLUMNS = ["gender", *MULTI_VALUE_COLUMNS]

SEPARATOR_PATTERN = re.compile(r"[、,，;；]+")


def split_values(value: object) -> list[str]:
    """Split a compound cell and remove blanks and duplicate tags."""
    if pd.isna(value):
        return []

    tags = [tag.strip() for tag in SEPARATOR_PATTERN.split(str(value))]
    return list(dict.fromkeys(tag for tag in tags if tag))


def parse_reasons(value: object, pfid: object) -> dict[str, str]:
    if pd.isna(value) or not str(value).strip():
        return {}

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"pfid={pfid} contains invalid reasons JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"pfid={pfid} reasons must be a JSON object")

    return {str(label).strip(): str(evidence).strip() for label, evidence in parsed.items()}


def prepare_data(source: Path, output_dir: Path) -> dict[str, int]:
    df = pd.read_csv(source, dtype={"pfid": "string"})

    required_columns = {"pfid", "reasons", *INDEXED_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df["pfid"].isna().any() or df["pfid"].duplicated().any():
        raise ValueError("pfid must be present and unique")

    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = df.copy()
    tag_rows: list[dict[str, str]] = []

    for column in INDEXED_COLUMNS:
        split_series = df[column].map(split_values)
        if column in MULTI_VALUE_COLUMNS:
            normalized[column] = split_series.map(
                lambda tags: json.dumps(tags, ensure_ascii=False)
            )

        for pfid, tags in zip(df["pfid"], split_series, strict=True):
            tag_rows.extend(
                {"pfid": pfid, "field": column, "value": tag} for tag in tags
            )

    reason_rows: list[dict[str, str]] = []
    normalized_reasons: list[str] = []
    for pfid, raw_reasons in zip(df["pfid"], df["reasons"], strict=True):
        reasons = parse_reasons(raw_reasons, pfid)
        normalized_reasons.append(json.dumps(reasons, ensure_ascii=False))
        reason_rows.extend(
            {"pfid": pfid, "label": label, "evidence": evidence}
            for label, evidence in reasons.items()
        )

    normalized["reasons"] = normalized_reasons
    tags_df = pd.DataFrame(tag_rows, columns=["pfid", "field", "value"])
    reasons_df = pd.DataFrame(
        reason_rows, columns=["pfid", "label", "evidence"]
    )

    normalized.to_csv(output_dir / "anchors_normalized.csv", index=False)
    tags_df.to_csv(output_dir / "anchor_tags.csv", index=False)
    reasons_df.to_csv(output_dir / "anchor_reasons.csv", index=False)

    inverted_index = build_inverted_index(tags_df)
    (output_dir / "tag_index.json").write_text(
        json.dumps(serialize_index(inverted_index), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "streamers": len(normalized),
        "tags": len(tags_df),
        "reasons": len(reasons_df),
        "index_keys": sum(len(values) for values in inverted_index.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("anchors_100.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_prepared"))
    args = parser.parse_args()

    counts = prepare_data(args.source, args.output_dir)
    print(
        "Prepared "
        f"{counts['streamers']} streamers, "
        f"{counts['tags']} tags, and "
        f"{counts['reasons']} reasons across "
        f"{counts['index_keys']} index keys in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
