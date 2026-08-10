"""Shared Streamlit cards for chat and pairwise recommendation results."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from recommender import matched_reason_evidence


CardMode = Literal["chat", "pairwise"]
CARD_HEIGHT = 620


RETRIEVAL_LABELS = {
    "exact": "類別標籤精確匹配",
    "literal_match": "姓名／帳號／PFID 逐字命中",
    "semantic_tiebreak": "標籤同分時以語意排序",
    "semantic_fallback": "純語意搜尋",
    "similar_streamer": "依所選主播尋找相似結果",
}


def _display_tags(tags: list[str]) -> str:
    return "、".join(tag.replace(":", "：") for tag in tags)


def _render_common_profile(row: object) -> None:
    vibe = str(row.overall_vibe).strip()
    st.write(f"{vibe[:77]}..." if len(vibe) > 80 else vibe)

    self_description = (
        str(row.self_description).strip()
        if pd.notna(row.self_description) and str(row.self_description).strip()
        else "尚未提供自我介紹"
    )
    if len(self_description) > 100:
        self_description = f"{self_description[:97]}..."
    st.markdown(f"**主播介紹**：{self_description}")


def _render_chat_details(row: object) -> None:
    st.caption(f"推薦來源：{RETRIEVAL_LABELS[row.retrieval_type]}")
    if row.matched_tags:
        st.caption(f"命中類別：{_display_tags(row.matched_tags)}")
    if row.matched_literals:
        st.caption(f"逐字命中：{'、'.join(row.matched_literals)}")

    reason_evidence = matched_reason_evidence(row.reasons, row.matched_tags)
    if reason_evidence:
        with st.expander("推薦理由"):
            for label, reason in reason_evidence:
                st.markdown(f"- **{label}**：{reason}")
    if row.retrieval_type != "exact":
        st.caption(f"語意相似度：{row.vector_score:.0%}")


def _render_pairwise_details(row: object) -> None:
    st.caption("推薦來源：二選一累積偏好訊號")
    if row.positive_matches:
        st.caption(f"偏好命中：{_display_tags(row.positive_matches[:4])}")

    reason_evidence = matched_reason_evidence(row.reasons, row.positive_matches)
    if reason_evidence:
        with st.expander("推薦理由"):
            for label, reason in reason_evidence:
                st.markdown(f"- **{label}**：{reason}")


def render_streamer_cards(
    recommendations: pd.DataFrame,
    mode: CardMode,
    enable_result_controls: bool = False,
    allow_similarity: bool = True,
) -> tuple[str, Literal["similar", "replace"]] | None:
    """Render up to five vertical streamer cards using a shared layout."""
    if recommendations.empty:
        return None
    result_action = None
    columns = st.columns(len(recommendations), gap="small")
    for column, row in zip(
        columns,
        recommendations.itertuples(index=False),
        strict=True,
    ):
        with column, st.container(border=True, height=CARD_HEIGHT):
            st.markdown(f"#### #{row.rank} 主播 {row.pfid}")
            if mode == "chat":
                if row.retrieval_type == "literal_match":
                    st.metric("逐字命中", f"{row.literal_score:.0%}")
                else:
                    st.metric("匹配度", f"{row.match_score:.0%}")
            else:
                st.metric("偏好分數", f"{row.pairwise_score:.2f}")
            _render_common_profile(row)
            if mode == "chat":
                _render_chat_details(row)
            else:
                _render_pairwise_details(row)
            st.caption(f"才藝：{row.talents}")
            st.caption(f"主題：{row.featured_topics}")
            st.caption(f"風格：{row.live_streaming_style}")
            if mode == "chat" and enable_result_controls:
                similar_column, replace_column = st.columns(2, gap="small")
                if similar_column.button(
                    "找相似",
                    key=f"find_similar_{row.pfid}",
                    help="保留這位並尋找四位相似主播",
                    width="stretch",
                    disabled=not allow_similarity,
                ):
                    result_action = (str(row.pfid), "similar")
                if replace_column.button(
                    "換一位",
                    key=f"replace_streamer_{row.pfid}",
                    help="只換掉這位主播",
                    width="stretch",
                ):
                    result_action = (str(row.pfid), "replace")
    return result_action
