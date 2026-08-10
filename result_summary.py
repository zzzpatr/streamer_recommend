"""Generate grounded LLM summaries for recommendation results."""

from __future__ import annotations

import json

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

from recommender import FIELD_LABELS, matched_reason_evidence


class RecommendationSummary(BaseModel):
    recommendation_reason: str
    selection_guide: str


def _compact_text(value: object, limit: int | None = None) -> str:
    text = " ".join(str(value).split()) if pd.notna(value) else ""
    return text[:limit] if limit else text


def _build_comparison_text(row: object) -> str:
    fields = [
        ("整體氛圍", row.overall_vibe),
        ("主播自述", _compact_text(row.self_description, 200)),
    ]
    return "；".join(
        f"{label}：{_compact_text(value)}"
        for label, value in fields
        if _compact_text(value)
    )


def build_summary_context(
    recommendations: pd.DataFrame,
    preferences: dict[str, list[str]],
    semantic_query: str,
    literal_queries: list[str],
) -> dict[str, object]:
    """Build a compact, JSON-safe evidence bundle for summary generation."""
    preference_labels = {
        FIELD_LABELS[field]: values
        for field, values in preferences.items()
        if field in FIELD_LABELS and values
    }
    candidates = []
    for row in recommendations.itertuples(index=False):
        reason_evidence = matched_reason_evidence(row.reasons, row.matched_tags)
        candidates.append(
            {
                "rank": int(row.rank),
                "pfid": str(row.pfid),
                "tag_score": round(float(row.tag_score), 4),
                "vector_score": round(float(row.vector_score), 4),
                "matched_tags": list(row.matched_tags),
                "matched_literals": list(row.matched_literals),
                "comparison_text": _build_comparison_text(row),
                "reason_evidence": [
                    {"label": label, "evidence": evidence}
                    for label, evidence in reason_evidence[:2]
                ],
            }
        )
    return {
        "preferences": preference_labels,
        "semantic_query": semantic_query,
        "literal_queries": literal_queries,
        "candidates": candidates,
    }


def build_pairwise_summary_context(
    results: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, object]:
    """Adapt a pairwise Top 5 snapshot to the shared summary context schema."""
    preferences = {field: [] for field in FIELD_LABELS}
    negative_signals = []
    for tag, weight in weights.items():
        if ":" not in tag:
            continue
        field, value = tag.split(":", 1)
        if field not in FIELD_LABELS:
            continue
        if weight > 0:
            preferences[field].append(value)
        elif weight < 0:
            negative_signals.append(
                {
                    "label": f"{FIELD_LABELS[field]}：{value}",
                    "weight": round(float(weight), 4),
                }
            )

    adapted = results.copy()
    adapted["tag_score"] = adapted["pairwise_score"]
    adapted["vector_score"] = 0.0
    adapted["matched_tags"] = adapted["positive_matches"]
    adapted["matched_literals"] = [[] for _ in range(len(adapted))]
    context = build_summary_context(adapted, preferences, "", [])
    context["selection_source"] = "pairwise_choices"
    context["negative_signals"] = sorted(
        negative_signals,
        key=lambda item: (item["weight"], item["label"]),
    )
    return context


def fallback_summary(context: dict[str, object]) -> dict[str, str]:
    """Return a short deterministic summary if the summary API is unavailable."""
    candidates = context.get("candidates", [])
    if not candidates:
        return {
            "recommendation_reason": "目前沒有足夠的候選主播可供比較。",
            "selection_guide": "請調整或增加偏好後再試一次。",
        }
    ids = "、".join(str(candidate["pfid"]) for candidate in candidates)
    if context.get("selection_source") == "similar_streamer":
        reason = (
            f"本次以主播 {context.get('anchor_pfid', '')} 的資料作為相似度基準，"
            f"選出 {ids}。"
        )
    elif context.get("selection_source") == "refreshed_results":
        reason = f"本次依照目前條件更新推薦組合，選出 {ids}。"
    else:
        reason = f"本次依照目前的偏好與主播資料，選出 {ids}。"
    return {
        "recommendation_reason": reason,
        "selection_guide": (
            "可以優先比較各主播的才藝、直播主題與互動風格，"
            "再從推薦卡片查看原始理由。"
        ),
    }


def generate_recommendation_summary(
    client: OpenAI,
    model: str,
    context: dict[str, object],
) -> dict[str, str]:
    """Generate a concise comparison using only supplied recommendation evidence."""
    response = client.responses.parse(
        model=model,
        instructions="""你是主播推薦結果摘要助手，請使用繁體中文。

根據輸入的偏好、五位候選 comparison_text 與 reason_evidence，填寫兩個欄位：
1. recommendation_reason：說明這批主播為什麼被推薦，指出主要命中條件。
2. selection_guide：比較五段 comparison_text，用 pfid 說明使用者可以怎麼選。

只能使用輸入資料，不可補充或猜測主播事實。reason_evidence 是標籤證據；comparison_text 中的主播自述只能描述為自述內容。
若 selection_source 是 similar_streamer，請說明這組結果以 anchor_pfid 的既有主播資料作為相似度基準，不要描述成使用者偏好或已驗證喜歡。
若 selection_source 是 refreshed_results，只能說明使用者更新了推薦組合，不可推測被換掉主播的負向特徵。
若 selection_source 是 pairwise_choices，請說明推薦來自使用者的二選一累積訊號；negative_signals 只是相對較弱偏好，不可描述成明確討厭或硬排除。
每個欄位只輸出一個純文字短段落，不要加入標題、清單、HTML、<br> 或換行控制符號。
""",
        input=json.dumps(context, ensure_ascii=False),
        text_format=RecommendationSummary,
        reasoning={"effort": "low"},
    )
    parsed = response.output_parsed
    if (
        parsed is None
        or not parsed.recommendation_reason.strip()
        or not parsed.selection_guide.strip()
    ):
        raise ValueError("模型未回傳可用的推薦摘要")
    return {
        "recommendation_reason": parsed.recommendation_reason.strip(),
        "selection_guide": parsed.selection_guide.strip(),
    }
