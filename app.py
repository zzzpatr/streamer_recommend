import hashlib
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI

from llm_service import MODEL, update_preference_state
from literal_search import search_literal_queries
from recommender import (
    FIELD_LABELS,
    build_ranking_query,
    load_streamers,
    load_tag_index,
    rank_streamers,
)
from preference_state import (
    empty_preferences,
    has_preferences,
    sanitize_literal_queries,
    sanitize_preferences,
)
from result_summary import (
    build_summary_context,
    fallback_summary,
    generate_recommendation_summary,
)
from result_controls import dismiss_current_batch, streamer_similarity_scores
from streamer_cards import render_streamer_cards
from vector_search import build_embedding_index_in_memory, semantic_search
from web_enrichment import (
    WebEnrichment,
    combine_semantic_query,
    expand_web_query,
)


TOP_N = 5


st.set_page_config(
    page_title="主播推薦助理",
    page_icon="💬",
    layout="wide",
)


def get_api_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except FileNotFoundError:
        return os.getenv("OPENAI_API_KEY")


def render_recommendation_summary(summary: object) -> None:
    """Render structured summary fields as two guaranteed paragraphs."""
    if isinstance(summary, dict):
        st.markdown(str(summary.get("recommendation_reason", "")))
        st.markdown(str(summary.get("selection_guide", "")))
    else:
        # Keep existing live sessions compatible with the old string schema.
        st.markdown(str(summary))


def reset_conversation() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "嗨！告訴我你喜歡的才藝、個性、主題、風格或感受。",
        }
    ]
    st.session_state.preferences = empty_preferences()
    st.session_state.excluded_preferences = empty_preferences()
    st.session_state.semantic_query = ""
    st.session_state.literal_queries = []
    st.session_state.web_enrichment = None
    st.session_state.web_enrichment_error = ""
    st.session_state.web_enrichment_cache = {}
    st.session_state.dismissed_pfids = set()
    st.session_state.similar_anchor_pfid = None
    st.session_state.latest_vector_scores = None
    st.session_state.latest_literal_scores = None
    st.session_state.latest_recommendations = pd.DataFrame()
    st.session_state.latest_recommendation_summary = {}
    st.session_state.pending_summary_context = None
    st.session_state.previous_response_id = None


def refresh_recommendations(
    streamers: pd.DataFrame,
    tag_index: object,
) -> None:
    """Rerank from cached retrieval scores without repeating API calls."""
    anchor_pfid = st.session_state.similar_anchor_pfid
    if anchor_pfid is not None:
        embedding_index = st.session_state.get("embedding_index")
        if embedding_index is None:
            st.session_state.similar_anchor_pfid = None
            anchor_pfid = None
        else:
            vector_scores = streamer_similarity_scores(
                embedding_index,
                anchor_pfid,
            )
    if anchor_pfid is None:
        vector_scores = st.session_state.latest_vector_scores

    if anchor_pfid is not None:
        include_anchor = anchor_pfid not in st.session_state.dismissed_pfids
        result_parts = []
        if include_anchor:
            result_parts.append(
                rank_streamers(
                    streamers=streamers,
                    tag_index=tag_index,
                    preferences=empty_preferences(),
                    excluded_preferences=st.session_state.excluded_preferences,
                    vector_scores=vector_scores[
                        vector_scores["pfid"] == anchor_pfid
                    ],
                    top_n=1,
                    semantic_threshold=0.0,
                )
            )
        similar_hidden = set(st.session_state.dismissed_pfids) | {anchor_pfid}
        result_parts.append(
            rank_streamers(
                streamers=streamers,
                tag_index=tag_index,
                preferences=empty_preferences(),
                excluded_preferences=st.session_state.excluded_preferences,
                vector_scores=vector_scores,
                hidden_pfids=similar_hidden,
                top_n=TOP_N - int(include_anchor),
                semantic_threshold=0.0,
            )
        )
        nonempty_parts = [part for part in result_parts if not part.empty]
        st.session_state.latest_recommendations = (
            pd.concat(nonempty_parts, ignore_index=True)
            if nonempty_parts
            else pd.DataFrame()
        )
        if not st.session_state.latest_recommendations.empty:
            st.session_state.latest_recommendations["rank"] = range(
                1, len(st.session_state.latest_recommendations) + 1
            )
            st.session_state.latest_recommendations["retrieval_type"] = (
                "similar_streamer"
            )
    else:
        st.session_state.latest_recommendations = rank_streamers(
            streamers=streamers,
            tag_index=tag_index,
            preferences=st.session_state.preferences,
            excluded_preferences=st.session_state.excluded_preferences,
            vector_scores=vector_scores,
            literal_scores=st.session_state.latest_literal_scores,
            hidden_pfids=st.session_state.dismissed_pfids,
            top_n=TOP_N,
        )
    semantic_query_for_ranking = combine_semantic_query(
        st.session_state.semantic_query,
        st.session_state.web_enrichment,
    )
    if st.session_state.latest_recommendations.empty:
        st.session_state.latest_recommendation_summary = {}
        st.session_state.pending_summary_context = None
        return
    st.session_state.latest_recommendation_summary = {}
    summary_context = build_summary_context(
        st.session_state.latest_recommendations,
        st.session_state.preferences,
        semantic_query_for_ranking,
        st.session_state.literal_queries,
    )
    if anchor_pfid is not None:
        summary_context["selection_source"] = "similar_streamer"
        summary_context["anchor_pfid"] = anchor_pfid
    elif st.session_state.dismissed_pfids:
        summary_context["selection_source"] = "refreshed_results"
    st.session_state.pending_summary_context = summary_context


api_key = get_api_key()

st.title("💬 主播推薦助理")
uploaded_file = st.file_uploader(
    "上傳主播 metadata CSV",
    type=["csv"],
    help="檔案只在目前的 Streamlit 執行階段中處理，不會寫入專案目錄。",
)
if uploaded_file is None:
    st.info("請先上傳 anchors_100.csv，再開始推薦對話。")
    st.stop()

file_bytes = uploaded_file.getvalue()
dataset_hash = hashlib.sha256(file_bytes).hexdigest()
try:
    streamers = load_streamers(BytesIO(file_bytes))
    tag_index = load_tag_index(streamers)
except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
    st.error(f"無法讀取 CSV：{exc}")
    st.stop()

# Share the validated dataset with the experimental multipage tools.
st.session_state.streamers = streamers

if st.session_state.get("dataset_hash") != dataset_hash:
    reset_conversation()
    st.session_state.dataset_hash = dataset_hash
    st.session_state.embedding_index = None

embedding_index = st.session_state.get("embedding_index")
embedding_index_error = None
if api_key and embedding_index is None:
    try:
        with st.spinner("正在為上傳資料建立語意索引…"):
            embedding_index = build_embedding_index_in_memory(
                OpenAI(api_key=api_key), streamers
            )
        st.session_state.embedding_index = embedding_index
    except Exception as exc:
        embedding_index_error = str(exc)

if "messages" not in st.session_state:
    reset_conversation()

# Keep live Streamlit sessions compatible after State schema changes.
for field in FIELD_LABELS:
    st.session_state.preferences.setdefault(field, [])
    st.session_state.excluded_preferences.setdefault(field, [])
if "semantic_query" not in st.session_state:
    st.session_state.semantic_query = ""
if "literal_queries" not in st.session_state:
    st.session_state.literal_queries = []
if "web_enrichment" not in st.session_state:
    st.session_state.web_enrichment = None
if "web_enrichment_error" not in st.session_state:
    st.session_state.web_enrichment_error = ""
if "web_enrichment_cache" not in st.session_state:
    st.session_state.web_enrichment_cache = {}
if "dismissed_pfids" not in st.session_state:
    st.session_state.dismissed_pfids = set()
if "similar_anchor_pfid" not in st.session_state:
    st.session_state.similar_anchor_pfid = None
if "latest_vector_scores" not in st.session_state:
    st.session_state.latest_vector_scores = None
if "latest_literal_scores" not in st.session_state:
    st.session_state.latest_literal_scores = None
if "latest_recommendation_summary" not in st.session_state:
    st.session_state.latest_recommendation_summary = {}
if "pending_summary_context" not in st.session_state:
    st.session_state.pending_summary_context = None

st.caption(
    f"{MODEL} 理解偏好｜類別精確匹配｜語意搜尋｜受控 Web 概念補充"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not api_key:
    st.error(
        "找不到 OPENAI_API_KEY。請將它設定在環境變數，或放入 "
        "`.streamlit/secrets.toml`。"
    )
else:
    prompt = st.chat_input(
        "例如：我想找像朋友陪在旁邊、讓人放鬆的女主播"
    )
    if prompt:
        # A new conversational query starts a fresh recommendation batch.
        st.session_state.dismissed_pfids = set()
        st.session_state.similar_anchor_pfid = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            client = OpenAI(api_key=api_key)
            
            with st.chat_message("assistant"):
                with st.spinner("更新偏好並搜尋主播中..."):
                    previous_semantic_query = st.session_state.semantic_query
                    turn, response_id = update_preference_state(
                        client=client,
                        tag_index=tag_index,
                        preferences=st.session_state.preferences,
                        excluded_preferences=(
                            st.session_state.excluded_preferences
                        ),
                        semantic_query=st.session_state.semantic_query,
                        literal_queries=st.session_state.literal_queries,
                        user_message=prompt,
                        previous_response_id=(
                            st.session_state.previous_response_id
                        ),
                    )
                    st.session_state.preferences = sanitize_preferences(
                        turn.preferences, tag_index
                    )
                    st.session_state.excluded_preferences = sanitize_preferences(
                        turn.excluded_preferences, tag_index
                    )
                    st.session_state.semantic_query = getattr(
                        turn, "semantic_query", ""
                    ).strip()
                    st.session_state.literal_queries = sanitize_literal_queries(
                        getattr(turn, "literal_queries", [])
                    )

                    if (
                        st.session_state.semantic_query
                        != previous_semantic_query
                    ):
                        st.session_state.web_enrichment = None
                        st.session_state.web_enrichment_error = ""

                    web_lookup_query = " ".join(
                        getattr(turn, "web_lookup_query", "").split()
                    )
                    if web_lookup_query:
                        try:
                            cached_enrichment = (
                                st.session_state.web_enrichment_cache.get(
                                    web_lookup_query
                                )
                            )
                            if cached_enrichment is None:
                                cached_enrichment = expand_web_query(
                                    client,
                                    MODEL,
                                    web_lookup_query,
                                )
                                st.session_state.web_enrichment_cache[
                                    web_lookup_query
                                ] = cached_enrichment
                            st.session_state.web_enrichment = cached_enrichment
                            st.session_state.web_enrichment_error = ""
                        except Exception as exc:
                            st.session_state.web_enrichment = None
                            st.session_state.web_enrichment_error = str(exc)

                    semantic_query_for_ranking = combine_semantic_query(
                        st.session_state.semantic_query,
                        st.session_state.web_enrichment,
                    )

                    ranking_query = build_ranking_query(
                        st.session_state.preferences,
                        semantic_query_for_ranking,
                    )
                    vector_scores = None
                    if ranking_query and embedding_index is not None:
                        vector_scores = semantic_search(
                            client,
                            ranking_query,
                            embedding_index,
                        )
                    literal_scores = search_literal_queries(
                        streamers,
                        st.session_state.literal_queries,
                    )
                    st.session_state.latest_vector_scores = vector_scores
                    st.session_state.latest_literal_scores = literal_scores
                    refresh_recommendations(streamers, tag_index)

                    st.markdown(turn.assistant_message)

            st.session_state.previous_response_id = response_id
            st.session_state.messages.append(
                {"role": "assistant", "content": turn.assistant_message}
            )
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"OpenAI API 呼叫失敗：{exc}")

with st.sidebar:
    st.subheader("目前偏好 State")
    has_any_state = False
    for field, label in FIELD_LABELS.items():
        wanted = st.session_state.preferences[field]
        excluded = st.session_state.excluded_preferences[field]
        if wanted or excluded:
            has_any_state = True
            st.markdown(f"**{label}**")
            if wanted:
                st.write("想要：" + "、".join(wanted))
            if excluded:
                st.write("排除：" + "、".join(excluded))

    if st.session_state.semantic_query:
        has_any_state = True
        st.markdown("**語意偏好**")
        st.write(st.session_state.semantic_query)

    enrichment: WebEnrichment | None = st.session_state.web_enrichment
    if enrichment is not None and enrichment.expanded_query:
        has_any_state = True
        st.markdown("**Web 概念補充**")
        st.write(enrichment.expanded_query)
        if enrichment.explanation:
            st.caption(enrichment.explanation)
        if enrichment.sources:
            source_links = "、".join(
                f"[{source.title.replace('[', '').replace(']', '')}]"
                f"(<{source.url}>)"
                for source in enrichment.sources
            )
            st.markdown(f"來源：{source_links}")

    if st.session_state.web_enrichment_error:
        st.warning("Web 概念補充失敗，本次仍使用原始語意推薦。")

    if st.session_state.literal_queries:
        has_any_state = True
        st.markdown("**逐字搜尋**")
        st.write("、".join(st.session_state.literal_queries))

    if not has_any_state:
        st.caption("尚未收集到偏好")

    if embedding_index_error:
        st.warning("語意索引尚未載入，超出類別的描述暫時無法搜尋")

    if st.button("清除對話與偏好", width="stretch"):
        reset_conversation()
        st.rerun()

st.divider()
recommendations = st.session_state.latest_recommendations
has_search_preference = (
    has_preferences(st.session_state.preferences)
    or bool(st.session_state.semantic_query)
    or bool(st.session_state.literal_queries)
)
summary_placeholder = None
if has_search_preference and not recommendations.empty:
    summary_placeholder = st.empty()
    if st.session_state.latest_recommendation_summary:
        with summary_placeholder.container():
            st.subheader("本次推薦摘要")
            render_recommendation_summary(
                st.session_state.latest_recommendation_summary
            )

st.subheader("最新推薦主播")

if not has_search_preference:
    st.info("在對話中告訴我你的偏好後，推薦結果會顯示在這裡。")
elif recommendations.empty:
    if st.session_state.dismissed_pfids:
        st.warning("這組條件下暫時沒有更多未看過的主播。")
        if st.button("恢復原推薦", type="primary"):
            st.session_state.dismissed_pfids = set()
            st.session_state.similar_anchor_pfid = None
            refresh_recommendations(streamers, tag_index)
            st.rerun()
    else:
        st.warning("目前條件沒有找到可推薦的主播，請調整或放寬偏好。")
else:
    control_left, control_middle, control_status = st.columns([1, 1, 4])
    if control_left.button("換一批", width="stretch"):
        st.session_state.dismissed_pfids = dismiss_current_batch(
            st.session_state.dismissed_pfids,
            recommendations["pfid"].astype(str).tolist(),
        )
        refresh_recommendations(streamers, tag_index)
        st.rerun()
    controls_active = bool(st.session_state.dismissed_pfids) or bool(
        st.session_state.similar_anchor_pfid
    )
    if control_middle.button(
        "恢復原推薦",
        width="stretch",
        disabled=not controls_active,
    ):
        st.session_state.dismissed_pfids = set()
        st.session_state.similar_anchor_pfid = None
        refresh_recommendations(streamers, tag_index)
        st.rerun()
    if st.session_state.similar_anchor_pfid:
        control_status.caption(
            "目前以主播 "
            f"{st.session_state.similar_anchor_pfid} 作為相似度基準"
        )
    elif st.session_state.dismissed_pfids:
        control_status.caption(
            f"本次已略過 {len(st.session_state.dismissed_pfids)} 位主播"
        )

    result_action = render_streamer_cards(
        recommendations,
        mode="chat",
        enable_result_controls=True,
        allow_similarity=embedding_index is not None,
    )
    if result_action is not None:
        pfid, action = result_action
        if action == "similar":
            st.session_state.similar_anchor_pfid = pfid
            st.session_state.dismissed_pfids = set()
        else:
            st.session_state.dismissed_pfids.add(pfid)
        refresh_recommendations(streamers, tag_index)
        st.rerun()

if (
    summary_placeholder is not None
    and st.session_state.pending_summary_context is not None
):
    summary_context = st.session_state.pending_summary_context
    with summary_placeholder.container():
        st.subheader("本次推薦摘要")
        with st.spinner("正在整理推薦摘要，主播結果已可先查看..."):
            try:
                summary_text = generate_recommendation_summary(
                    OpenAI(api_key=api_key),
                    MODEL,
                    summary_context,
                )
            except Exception:
                summary_text = fallback_summary(summary_context)
        render_recommendation_summary(summary_text)
    st.session_state.latest_recommendation_summary = summary_text
    st.session_state.pending_summary_context = None
