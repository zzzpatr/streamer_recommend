"""OpenAI preference-state extraction for the conversational recommender."""

from __future__ import annotations

import json

from openai import OpenAI

from preference_state import ConversationTurn
from search_index import TagIndex


MODEL = "gpt-5.6-luna"


def build_instructions(tag_index: TagIndex) -> str:
    vocabulary = {
        field: sorted(values)
        for field, values in tag_index.items()
    }
    return f"""你是主播推薦系統的偏好解析 Agent，請使用繁體中文。

每輪根據目前 State 和使用者最新訊息，輸出更新後的完整 State：
- preferences 是使用者想要的標準類別條件。
- excluded_preferences 是使用者明確不想要的標準類別條件。
- semantic_query 只保存無法可靠映射到標準類別、但對推薦有意義的描述。
- literal_queries 只保存主播姓名、暱稱、社群帳號或 pfid，用於本地主播身份逐字搜尋。
- web_lookup_query 只在使用者最新訊息明確提到需要外部知識才能理解的作品、角色、IP、遊戲或流行語時，填入適合搜尋的簡短查詢；其他情況必須留空。
- 使用者沒有修改的既有條件、semantic_query 與 literal_queries 必須保留。
- 明確的「不要／排除／不想看」要從 preferences 移除並加入 excluded_preferences。
- 「取消偏好／不一定要」只代表放寬條件，不要加入 excluded_preferences。
- 若使用者改變主意，正確加入、移除或取代條件。
- 類別只能逐字使用下方標準標籤，不可拆字或創造標籤。
- 將明確近義詞映射到最接近的標準標籤。
- 已被標準標籤完整表達的內容不要放入 semantic_query。例如「啦啦隊」能映射為「啦啦隊應援」，semantic_query 應留空。
- 無法類別化的感受或抽象需求，例如「像朋友陪伴、讓人安心」，放入 semantic_query。
- 「名字／名子是 X」、「叫 X」、「名為 X」、「帳號是 X」表示要找包含 X 的主播，將 X 原樣加入 literal_queries。
- 若使用者最新訊息去除前後空白後完全由數字組成，一律將完整數字視為主播 PFID，原樣加入 literal_queries；不要追問它是不是帳號或 PFID。
- 「找 4475935」、「主播 4475935」、「PFID 是 4475935」等包含明確數字識別碼的搜尋，也要將數字原樣加入 literal_queries。
- PFID 是否存在由後續逐字搜尋驗證；即使不確定或可能查無資料，也不可因此拒絕加入 literal_queries 或先向使用者追問。
- 作品、動漫角色、IP、遊戲、流行語與風格概念不是主播身份，不可放入 literal_queries；例如「像芙莉蓮」、「寶可夢風格」應原樣放入 semantic_query。
- 「像芙莉蓮」可將 web_lookup_query 設為「芙莉蓮 動漫角色 性格 特質」；普通形容詞、標準標籤、主播姓名、PFID，以及「恐龍」這類用途不明的普通名詞不可觸發 Web Search。
- web_lookup_query 只反映使用者最新訊息，不可沿用上一輪的值；外部搜尋結果由後續程式處理，不可自行捏造或寫回 preferences。
- literal_queries 必須保留原始拼字，不可翻譯或改寫；新的值應加入既有清單，只有使用者明確取消或更換時才移除。
- 不要將 literal_queries 同時放入 semantic_query。只有使用者明確說「我叫 X」時，才把它理解成使用者自我介紹而非主播搜尋。
- 只有 preferences、semantic_query、literal_queries 都沒有新增或既有內容，而且最新訊息也不構成可用搜尋時，才用 assistant_message 簡短追問。
- assistant_message 只確認理解，不捏造或直接推薦主播。

標準標籤：
{json.dumps(vocabulary, ensure_ascii=False)}
"""


def update_preference_state(
    client: OpenAI,
    tag_index: TagIndex,
    preferences: dict[str, list[str]],
    excluded_preferences: dict[str, list[str]],
    semantic_query: str,
    literal_queries: list[str],
    user_message: str,
    previous_response_id: str | None = None,
) -> tuple[ConversationTurn, str]:
    current_state = {
        "preferences": preferences,
        "excluded_preferences": excluded_preferences,
        "semantic_query": semantic_query,
        "literal_queries": literal_queries,
    }
    request = {
        "model": MODEL,
        "instructions": build_instructions(tag_index),
        "input": (
            f"目前 State：{json.dumps(current_state, ensure_ascii=False)}\n\n"
            f"使用者最新訊息：{user_message}"
        ),
        "text_format": ConversationTurn,
        "reasoning": {"effort": "low"},
    }
    if previous_response_id:
        request["previous_response_id"] = previous_response_id

    response = client.responses.parse(**request)
    turn = response.output_parsed
    if turn is None:
        raise ValueError("模型未回傳可解析的偏好 State")
    return turn, response.id
