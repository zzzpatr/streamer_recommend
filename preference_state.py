"""Schemas and validation helpers for conversational preference state."""

from __future__ import annotations

from pydantic import BaseModel

from search_index import TagIndex


PREFERENCE_FIELDS = (
    "gender",
    "personality",
    "appearance",
    "talents",
    "featured_topics",
    "live_streaming_style",
)


class PreferenceValues(BaseModel):
    gender: list[str]
    personality: list[str]
    appearance: list[str]
    talents: list[str]
    featured_topics: list[str]
    live_streaming_style: list[str]


class ConversationTurn(BaseModel):
    assistant_message: str
    preferences: PreferenceValues
    excluded_preferences: PreferenceValues
    semantic_query: str
    literal_queries: list[str]
    web_lookup_query: str


def empty_preferences() -> dict[str, list[str]]:
    return {field: [] for field in PREFERENCE_FIELDS}


def sanitize_preferences(
    preferences: PreferenceValues | dict[str, list[str]],
    index: TagIndex,
) -> dict[str, list[str]]:
    """Keep only canonical values available in the local tag index."""
    raw = (
        preferences.model_dump()
        if isinstance(preferences, PreferenceValues)
        else preferences
    )
    return {
        field: list(
            dict.fromkeys(
                value
                for value in raw.get(field, [])
                if value in index.get(field, {})
            )
        )
        for field in PREFERENCE_FIELDS
    }


def has_preferences(preferences: dict[str, list[str]]) -> bool:
    return any(preferences.values())


def sanitize_literal_queries(queries: list[str]) -> list[str]:
    """Normalize and deduplicate literal searches while preserving their text."""
    cleaned = (" ".join(str(query).split()) for query in queries)
    return list(dict.fromkeys(query for query in cleaned if query))[:10]
