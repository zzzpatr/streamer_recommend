"""Experimental Streamlit page for pairwise streamer preference exploration."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI

from llm_service import MODEL
from pairwise import (
    build_profile_tags,
    choose_pair,
    rank_profiles,
    strongest_signals,
    update_weights,
)
from recommender import FIELD_LABELS, load_streamers
from result_summary import (
    build_pairwise_summary_context,
    fallback_summary,
    generate_recommendation_summary,
)
from streamer_cards import render_streamer_cards


RESULT_INTERVAL = 5
CHOICE_CARD_HEIGHT = 430


st.set_page_config(
    page_title="Pairwise Streamer Explorer",
    page_icon="⚖️",
    layout="wide",
)


def get_api_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except FileNotFoundError:
        return os.getenv("OPENAI_API_KEY")


def display_tag(tag: str) -> str:
    field, value = tag.split(":", 1)
    return f"{FIELD_LABELS.get(field, field)}：{value}"


def render_summary(summary: dict[str, str]) -> None:
    st.markdown(summary.get("recommendation_reason", ""))
    st.markdown(summary.get("selection_guide", ""))


def reset_pairwise(dataset_hash: str, profiles: dict[str, frozenset[str]]) -> None:
    st.session_state.pw_dataset_hash = dataset_hash
    st.session_state.pw_weights = {}
    st.session_state.pw_history = []
    st.session_state.pw_seen_pfids = set()
    st.session_state.pw_current_pair = choose_pair(profiles, {}, set())
    st.session_state.pw_result_round = 0
    st.session_state.pw_results = pd.DataFrame()
    st.session_state.pw_summary = {}
    st.session_state.pw_pending_summary_context = None


def select_streamer(
    winner: str,
    loser: str,
    profiles: dict[str, frozenset[str]],
) -> None:
    st.session_state.pw_weights = update_weights(
        st.session_state.pw_weights,
        profiles[winner],
        profiles[loser],
    )
    st.session_state.pw_history.append({"winner": winner, "loser": loser})
    st.session_state.pw_seen_pfids.update((winner, loser))
    st.session_state.pw_current_pair = choose_pair(
        profiles,
        st.session_state.pw_weights,
        st.session_state.pw_seen_pfids,
    )
    st.rerun()


def render_profile(row: pd.Series) -> None:
    st.write(row["overall_vibe"])
    st.caption(f"性別：{row['gender']}｜性格：{row['personality']}")
    st.caption(f"才藝：{row['talents']}")
    st.caption(f"主題：{row['featured_topics']}")
    st.caption(f"風格：{row['live_streaming_style']}")
    description = " ".join(str(row["self_description"]).split())
    if description:
        st.markdown(
            f"**主播自述：** {description[:177]}..."
            if len(description) > 180
            else f"**主播自述：** {description}"
        )


def build_result_snapshot(
    streamers: pd.DataFrame,
    profiles: dict[str, frozenset[str]],
    weights: dict[str, float],
) -> pd.DataFrame:
    ranked = rank_profiles(profiles, weights).head(5)
    results = ranked.merge(streamers, on="pfid", how="left", validate="one_to_one")
    results.insert(0, "rank", range(1, len(results) + 1))
    return results


st.title("⚖️ 主播二選一偏好探索")
st.write(
    "每輪選一位比較喜歡的主播。探索沒有總輪數限制；系統每完成五輪，"
    "就依目前累積的弱偏好訊號更新一次 Top 5 與推薦摘要。"
)

streamers = st.session_state.get("streamers")
dataset_hash = st.session_state.get("dataset_hash")

if streamers is None:
    uploaded_file = st.file_uploader(
        "尚未從主頁取得資料，請在此上傳主播 metadata CSV",
        type=["csv"],
    )
    if uploaded_file is None:
        st.info("請先到主頁上傳 anchors_100.csv，或直接在此頁上傳。")
        st.stop()
    file_bytes = uploaded_file.getvalue()
    dataset_hash = hashlib.sha256(file_bytes).hexdigest()
    try:
        streamers = load_streamers(BytesIO(file_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"無法讀取 CSV：{exc}")
        st.stop()
    st.session_state.streamers = streamers
    st.session_state.dataset_hash = dataset_hash

profiles = build_profile_tags(streamers)
if len(profiles) < 2:
    st.error("二選一至少需要兩位主播。")
    st.stop()

if (
    st.session_state.get("pw_dataset_hash") != dataset_hash
    or "pw_result_round" not in st.session_state
):
    reset_pairwise(dataset_hash, profiles)

completed_rounds = len(st.session_state.pw_history)
positive, negative = strongest_signals(st.session_state.pw_weights)
with st.sidebar:
    st.subheader("目前推測偏好")
    st.caption(f"已完成 {completed_rounds} 輪選擇")
    if positive:
        st.markdown("**較偏好的特徵**")
        st.write("、".join(display_tag(tag) for tag, _ in positive))
    if negative:
        st.markdown("**相對較不偏好的特徵**")
        st.write("、".join(display_tag(tag) for tag, _ in negative))
    if not positive and not negative:
        st.caption("完成第一輪選擇後，推測特徵會顯示在這裡。")

milestone_round = completed_rounds // RESULT_INTERVAL * RESULT_INTERVAL
if milestone_round >= RESULT_INTERVAL and milestone_round > st.session_state.pw_result_round:
    results = build_result_snapshot(
        streamers,
        profiles,
        st.session_state.pw_weights,
    )
    st.session_state.pw_result_round = milestone_round
    st.session_state.pw_results = results
    st.session_state.pw_summary = {}
    st.session_state.pw_pending_summary_context = build_pairwise_summary_context(
        results,
        st.session_state.pw_weights,
    )

rounds_since_update = completed_rounds % RESULT_INTERVAL
rounds_until_update = RESULT_INTERVAL - rounds_since_update
toolbar_left, toolbar_right = st.columns([1, 4])
with toolbar_left:
    if st.button("重新開始", width="stretch"):
        reset_pairwise(dataset_hash, profiles)
        st.rerun()
with toolbar_right:
    if completed_rounds and rounds_since_update == 0:
        progress_text = (
            f"已完成 {completed_rounds} 輪，Top 5 已更新；"
            f"再選 {RESULT_INTERVAL} 輪會更新下一版"
        )
    else:
        progress_text = (
            f"已完成 {completed_rounds} 輪；再選 {rounds_until_update} 輪更新 Top 5"
        )
    st.progress(rounds_since_update / RESULT_INTERVAL, text=progress_text)

left_id, right_id = st.session_state.pw_current_pair
indexed = streamers.set_index("pfid")
left_column, right_column = st.columns(2, gap="large")

with left_column, st.container(border=True, height=CHOICE_CARD_HEIGHT):
    st.markdown(f"### 主播 {left_id}")
    if st.button("選這位主播", key=f"choose_{completed_rounds}_{left_id}", width="stretch"):
        select_streamer(left_id, right_id, profiles)
    render_profile(indexed.loc[left_id])

with right_column, st.container(border=True, height=CHOICE_CARD_HEIGHT):
    st.markdown(f"### 主播 {right_id}")
    if st.button("選這位主播", key=f"choose_{completed_rounds}_{right_id}", width="stretch"):
        select_streamer(right_id, left_id, profiles)
    render_profile(indexed.loc[right_id])

if st.session_state.pw_result_round:
    st.divider()
    st.caption(f"以下結果依前 {st.session_state.pw_result_round} 輪選擇產生")
    summary_placeholder = st.empty()
    if st.session_state.pw_summary:
        with summary_placeholder.container():
            st.subheader("二選一推薦摘要")
            render_summary(st.session_state.pw_summary)

    st.subheader("二選一推薦 Top 5")
    render_streamer_cards(st.session_state.pw_results, mode="pairwise")

    if st.session_state.pw_pending_summary_context is not None:
        context = st.session_state.pw_pending_summary_context
        api_key = get_api_key()
        with summary_placeholder.container():
            st.subheader("二選一推薦摘要")
            if api_key:
                with st.spinner("正在整理二選一推薦摘要，Top 5 已可先查看..."):
                    try:
                        summary = generate_recommendation_summary(
                            OpenAI(api_key=api_key),
                            MODEL,
                            context,
                        )
                    except Exception:
                        summary = fallback_summary(context)
            else:
                summary = fallback_summary(context)
            render_summary(summary)
        st.session_state.pw_summary = summary
        st.session_state.pw_pending_summary_context = None

    with st.expander("查看全部選擇紀錄"):
        for index, choice in enumerate(st.session_state.pw_history, start=1):
            st.write(
                f"第 {index} 輪：選擇 {choice['winner']}，"
                f"未選 {choice['loser']}"
            )
