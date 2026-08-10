"""Build and query a small local OpenAI embedding index for streamers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
ARTIFACT_NAME = "streamer_embeddings.npz"
METADATA_NAME = "embedding_metadata.json"
DOCUMENTS_NAME = "streamer_documents.json"


def build_streamer_document(row: pd.Series) -> str:
    reasons = row.get("reasons", "")
    try:
        parsed_reasons = json.loads(reasons) if pd.notna(reasons) else {}
        evidence = "；".join(
            f"{label}：{text}" for label, text in parsed_reasons.items()
        )
    except (json.JSONDecodeError, TypeError):
        evidence = str(reasons) if pd.notna(reasons) else ""

    fields = [
        ("性別", row.get("gender")),
        ("性格", row.get("personality")),
        ("外型", row.get("appearance")),
        ("才藝", row.get("talents")),
        ("直播主題", row.get("featured_topics")),
        ("直播風格", row.get("live_streaming_style")),
        ("整體氛圍", row.get("overall_vibe")),
        ("自我介紹", row.get("self_description")),
        ("標籤判斷依據", evidence),
    ]
    return "\n".join(
        f"{label}：{value}"
        for label, value in fields
        if pd.notna(value) and str(value).strip()
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def build_embedding_index(
    client: OpenAI,
    streamers: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 64,
) -> None:
    documents = [build_streamer_document(row) for _, row in streamers.iterrows()]
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        embeddings.extend(item.embedding for item in response.data)

    matrix = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    pfids = streamers["pfid"].astype("string").to_numpy(dtype=str)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / ARTIFACT_NAME, pfids=pfids, embeddings=matrix)

    document_records = [
        {"pfid": pfid, "document": document}
        for pfid, document in zip(pfids.tolist(), documents, strict=True)
    ]
    (output_dir / DOCUMENTS_NAME).write_text(
        json.dumps(document_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "model": EMBEDDING_MODEL,
        "dimensions": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "created_at": datetime.now(UTC).isoformat(),
        "documents_sha256": hashlib.sha256(
            json.dumps(document_records, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_embedding_index_in_memory(
    client: OpenAI,
    streamers: pd.DataFrame,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create an embedding index without persisting uploaded data to disk."""
    documents = [build_streamer_document(row) for _, row in streamers.iterrows()]
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=documents[start : start + batch_size],
            encoding_format="float",
        )
        embeddings.extend(item.embedding for item in response.data)

    matrix = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    pfids = streamers["pfid"].astype("string").to_numpy(dtype=str)
    metadata = {
        "model": EMBEDDING_MODEL,
        "dimensions": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
    }
    return pfids, matrix, metadata


def load_embedding_index(output_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    artifact = output_dir / ARTIFACT_NAME
    metadata_path = output_dir / METADATA_NAME
    if not artifact.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Embedding index does not exist. Run `python vector_search.py --build`."
        )
    with np.load(artifact, allow_pickle=False) as data:
        pfids = data["pfids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(pfids) != len(embeddings) or len(pfids) != metadata["count"]:
        raise ValueError("Embedding artifact and metadata counts do not match")
    return pfids, embeddings, metadata


def semantic_search(
    client: OpenAI,
    query: str,
    embedding_index: tuple[np.ndarray, np.ndarray, dict],
) -> pd.DataFrame:
    pfids, embeddings, metadata = embedding_index
    response = client.embeddings.create(
        model=metadata["model"],
        input=[query],
        encoding_format="float",
    )
    query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm:
        query_vector /= query_norm
    cosine_scores = embeddings @ query_vector
    # Negative similarity is not useful for recommendation ranking.
    vector_scores = np.clip(cosine_scores, 0.0, 1.0)
    return pd.DataFrame({"pfid": pfids, "vector_score": vector_scores})


def _load_cli_api_key(base_dir: Path) -> str:
    key = os.getenv("OPENAI_API_KEY")
    secrets_path = base_dir / ".streamlit" / "secrets.toml"
    if not key and secrets_path.exists():
        key = tomllib.loads(secrets_path.read_text(encoding="utf-8")).get(
            "OPENAI_API_KEY"
        )
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the index")
    parser.add_argument("--source", type=Path, default=Path("anchors_100.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_prepared"))
    args = parser.parse_args()
    if not args.build:
        parser.error("Specify --build")

    base_dir = Path.cwd()
    streamers = pd.read_csv(args.source, dtype={"pfid": "string"})
    client = OpenAI(api_key=_load_cli_api_key(base_dir))
    build_embedding_index(client, streamers, args.output_dir)
    _, matrix, metadata = load_embedding_index(args.output_dir)
    print(
        f"Built {metadata['count']} embeddings with {metadata['model']} "
        f"({matrix.shape[1]} dimensions)."
    )


if __name__ == "__main__":
    main()
