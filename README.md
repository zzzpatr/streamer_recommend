# 主播推薦助理

以 131 位主播 metadata 實作的對話式推薦 Demo，對應 take-home assignment 的「主播推薦系統」。使用者可用自然語言描述偏好、排除特徵，或輸入主播姓名、帳號與 PFID；系統會以標籤、逐字搜尋和 Embedding 排出 Top 5，並使用原始 `reasons` 佐證推薦結果。作品、角色、遊戲或流行語可透過受控 Web Search 補充語意特徵。

選擇此題是因為資料同時具有多值標籤，以及 `overall_vibe`、`self_description`、`reasons` 等敘述文字，適合展示 Hybrid Retrieval、可解釋排序與 grounded generation。

## 系統流程

```mermaid
flowchart TD
    A[Input<br/>自然語言或主播身份] --> B[Data Prepared<br/>驗證、標籤索引、Embedding]
    B --> C[LLM Structure<br/>更新 Preference State]
    C --> W{需要外部概念知識?}
    W -- 是 --> X[Web Search<br/>只擴充語意查詢]
    W -- 否 --> D[Reasoning<br/>召回、評分、排除、排序]
    X --> D
    D --> E[Output<br/>Top 5、證據與摘要]
    E --> H[Result Controls<br/>換一位、換一批、找相似]
    H --> D
    E --> F[Feedback Prototype<br/>獨立二選一偏好探索]
    F --> G[Pairwise Top 5]
```

## 1. Input

目前支援兩種入口。

### 偏好推薦

使用自然語言描述想要或不想要的條件：

```text
我想找會唱歌、親切友善的男主播
想找像朋友一樣陪伴、讓人放鬆的主播
想找像芙莉蓮一樣的主播
不要互動太熱絡，偏好音樂陪伴
改成性別不限，但保留歌唱偏好
```

支援多輪新增、修改、取消與排除偏好。未修改的 State 會保留至後續對話。

### 主播身份搜尋

姓名、暱稱、社群帳號與 PFID 使用本地逐字搜尋：

```text
名字是 Eason Lee
4475935
PFID 是 4475935
```

搜尋不分英文大小寫並支援子字串。純數字輸入一律先視為 PFID；目前不支援錯字修正、fuzzy matching 或同名消歧。

## 2. Data Prepared

使用者在 Streamlit 上傳 CSV，必要欄位如下：

| 欄位 | 用途 |
| --- | --- |
| `pfid` | 主播唯一 ID 與身份搜尋 |
| `gender` | 性別軟偏好 |
| `personality` | 性格定位 |
| `appearance` | 外型特徵 |
| `talents` | 才藝 |
| `featured_topics` | 直播主題 |
| `live_streaming_style` | 直播互動與內容風格 |
| `overall_vibe` | 語意搜尋與摘要比較 |
| `reasons` | 標籤的 JSON 原始證據 |
| `self_description` | 語意搜尋、身份搜尋與摘要比較 |

處理流程：

1. 驗證必要欄位、空白 `pfid` 與重複 `pfid`。
2. 將頓號、半形／全形逗號與分號分隔的多值欄位正規化。
3. 建立 `field → value → pfid set` 記憶體倒排索引。
4. 將每位主播的結構化欄位、整體氛圍、自我介紹和 `reasons` 組成語意文件。
5. 使用 `text-embedding-3-small` 建立主播向量並保存在目前 Session 記憶體。

`reasons` key 的主要格式：

```text
男性／女性
純真可愛型
外型_可愛甜美
才藝_歌唱
主題_日常輕鬆閒聊
風格_互動熱絡
```

性別欄位的 `男／女` 會對應到 reason key 的 `男性／女性`。其他欄位優先查找「中文前綴_標籤值」，找不到時再以純標籤值 fallback；原始資料沒有證據時不生成理由。

線上 Demo 不需預先執行 `prepare_data.py` 或 `vector_search.py --build`，兩者僅保留為本機離線分析工具。

## 3. LLM Structure

LLM 不直接決定推薦誰，只負責將每輪訊息更新為完整 Preference State：

```json
{
  "assistant_message": "已更新你的主播偏好。",
  "preferences": {
    "gender": ["男"],
    "personality": [],
    "appearance": [],
    "talents": ["歌唱"],
    "featured_topics": [],
    "live_streaming_style": ["親切友善"]
  },
  "excluded_preferences": {
    "gender": [],
    "personality": [],
    "appearance": [],
    "talents": [],
    "featured_topics": [],
    "live_streaming_style": []
  },
  "semantic_query": "像朋友一樣陪伴",
  "literal_queries": ["Eason Lee"],
  "web_lookup_query": "芙莉蓮 動漫角色 性格 特質"
}
```

分流規則：

```text
能映射的標準標籤       → preferences
明確不要的標準標籤     → excluded_preferences
抽象感受、作品或角色概念 → semantic_query
主播姓名、帳號、PFID    → literal_queries
需外部知識的明確實體    → web_lookup_query
```

格式由 Pydantic Structured Outputs 約束。LLM 回傳後：

- `sanitize_preferences()` 只保留目前 CSV 倒排索引中存在的標籤。
- `sanitize_literal_queries()` 移除空白與重複值，最多保留 10 筆。
- `previous_response_id` 與目前完整 State 用來延續多輪對話。
- `web_lookup_query` 只反映最新一輪，不會累積進 State；普通形容詞、標準標籤、主播姓名、PFID 與用途不明的普通名詞不會觸發搜尋。

## 4. Reasoning

系統採用逐字搜尋、倒排索引與 Embedding 的 Hybrid Retrieval。

### 4.1 身份逐字搜尋

`literal_queries` 只搜尋：

```text
pfid
self_description
```

逐字命中代表使用者正在尋找特定主播，因此排序優先於一般偏好分數。

### 4.2 標籤加權

所有正向條件包含性別，都是軟偏好：

| 欄位 | 原始權重 |
| --- | ---: |
| 性別 | 0.15 |
| 性格 | 0.15 |
| 外型 | 0.15 |
| 才藝 | 0.25 |
| 直播主題 | 0.20 |
| 直播風格 | 0.25 |

只針對實際啟用欄位重新正規化：

```text
normalized_weight(field) = field_weight / sum(active_field_weights)
score_per_tag = normalized_weight(field) / number_of_values_in_field
tag_score = sum(score_per_tag for matched tags)
```

同欄位多個值會平均分配該欄位權重。`excluded_preferences` 是硬排除，命中任一排除標籤即移除候選。

### 4.3 受控 Web query expansion

若最新訊息明確包含需要外部知識才能理解的作品、角色、IP、遊戲或流行語，系統才會額外呼叫 Responses API：

```python
tools=[{"type": "web_search"}]
```

Web Search 將實體整理成人格、互動方式、內容主題、才藝、視覺風格或整體氛圍等短特徵。結果只與原始 `semantic_query` 合併後送進 Embedding，不會寫入 `preferences`、修改主播 metadata 或直接決定候選。相同 lookup query 在目前 Session 中會快取；搜尋失敗或內容不明確時退回原始語意查詢。

介面會在語意補充旁顯示最多三個可點擊來源。外部來源僅用於理解使用者所指的概念，主播事實與推薦證據仍只取自上傳 CSV。

### 4.4 語意分數

結構化偏好與 `semantic_query` 會組成完整查詢並建立 query vector：

```text
vector_score = clip(normalized_streamer_vector · normalized_query_vector, 0, 1)
```

這是 cosine similarity，不是命中機率。

### 4.5 最終排名

```text
literal_score DESC
tag_score DESC
vector_score DESC
pfid ASC
```

- `tag_score` 不同時，語意分數不能推翻較高的標籤分數。
- `tag_score` 相同時，`vector_score` 作為 tie-breaker。
- 沒有逐字或標籤候選時，純語意結果需達 `0.20` 門檻。
- 向量搜尋不會補滿不足 Top 5 的標籤結果。
- `pfid` 只用於最後的穩定排序，不代表主播品質。

## 5. Output

Streamlit 會顯示：

- Sidebar 中目前累積的正向偏好、排除偏好、語意偏好、Web 概念補充與逐字搜尋。
- Top 5 固定等高的直式主播卡片；內容超出時在卡片內捲動，避免自我介紹長短影響版面。
- 匹配度、推薦來源、命中標籤與語意相似度。
- 才藝、直播主題、直播風格、整體氛圍與截短的主播自我介紹。
- 可展開的 `reasons` 原始推薦證據。
- 「為什麼推薦」及「五位怎麼選」兩段式 LLM 摘要。
- 每張聊天推薦卡片的「找相似／換一位」，以及整組「換一批」。

### 推薦摘要

推薦排名完成後會先渲染五張主播卡片，再呼叫 LLM 產生摘要，完成後填回卡片上方的預留位置。

摘要 context 包含：

- 目前偏好與查詢。
- 候選分數、命中標籤及逐字命中。
- 若由「找相似」或更新組合產生，包含對應的結果來源與基準 PFID。
- 每位主播最多兩筆 `reason_evidence`。
- 由 `overall_vibe` 和前 200 字 `self_description` 組成的 `comparison_text`。

摘要使用兩欄 Structured Output，Streamlit 分別渲染，避免依賴 `<br>` 或 Markdown 空行：

```json
{
  "recommendation_reason": "為什麼推薦這批主播。",
  "selection_guide": "五位主播可以怎麼選。"
}
```

`self_description` 屬於主播自述，不等同已驗證事實；`reasons` 是標籤命中的主要證據。摘要 API 失敗時改用規則式 fallback，不影響已完成的推薦卡片。

## 6. Feedback

### 聊天推薦結果控制

聊天頁不要求使用者替系統標註喜歡或不喜歡的特徵，而是提供具有立即結果的操作：

| 操作 | 行為 |
| --- | --- |
| `換一位` | 本次查詢略過該 PFID，以原排名的下一順位補上 |
| `換一批` | 略過目前五位，顯示下一組候選 |
| `找相似` | 保留所選主播作為向量基準，顯示該主播與四位相似主播 |
| `恢復原推薦` | 清除本次略過清單與相似基準，回到原始排序 |

`換一位` 和 `換一批` 只更新 Session 內的 `dismissed_pfids`，不推測負向特徵，也不修改 Preference State。輸入新的聊天訊息時會清除本次略過狀態。

`找相似` 直接取所選主播已建立的向量，與全部主播向量計算 cosine similarity；因此會一起參考結構化欄位、`overall_vibe`、`self_description` 與 `reasons`，不需要重新呼叫 Embedding API。明確的 `excluded_preferences` 仍會套用，但原本正向條件不會壓過相似度排序。

### 二選一偏好探索

目前提供獨立的「主播二選一偏好探索」Streamlit 分頁，用來測試使用者不知道如何描述偏好時的迭代式推薦。

每輪顯示兩位差異較大的主播。使用者選擇其中一位後：

```text
選中主播獨有標籤：本輪合計 +1.0
未選主播獨有標籤：本輪合計 -0.25
兩位共同標籤：不更新
```

加減分會平均分配到該輪的差異標籤，因此標籤較多的主播不會僅因欄位較豐富而取得更多總更新。未選特徵只是弱負向訊號，不會成為硬排除。

第一輪使用 Jaccard distance 選擇差異較大的兩位，以增加探索資訊；之後從目前較高分候選中挑選特徵差異較大的對手。探索沒有總輪數限制，每完成第 5、10、15…輪就更新一次結果快照：

- 推測出的較偏好與相對較不偏好特徵。
- Pairwise score 排序的 Top 5。
- 與聊天推薦頁共用版型的五張主播卡片，包含自我介紹、才藝、主題、風格與可展開的 `reasons`。
- 沿用 `result_summary.py` 產生的兩段式推薦摘要。
- 每輪選擇紀錄。

推測出的正向與弱負向特徵會隨每輪選擇即時顯示在 Pairwise Explorer 的 Sidebar；主畫面保留二選一、推薦摘要與 Top 5。

在兩個里程碑之間會保留上一版 Top 5，使用者可以繼續選擇；達到下一個五輪倍數時才重新排名與生成摘要。摘要使用 pairwise 正負訊號、Top 5 metadata 與 `reasons`，未選特徵只會描述成相對較弱偏好。

這個實驗頁不會修改聊天推薦的 Preference State，方便獨立比較兩種互動模式。主頁上傳過 CSV 後可跨頁共用；也能直接在二選一頁上傳資料。沒有 API Key 時仍可完成二選一排名，摘要改用規則式 fallback。

目前尚未實作跨 Session 使用者檔案或以真實行為資料訓練的 learning-to-rank。後續可加入：

- Pairwise preference data。
- Precision@K、NDCG@K、換一位率與相似結果點擊率。

## 專案模組

| 檔案 | 職責 |
| --- | --- |
| `app.py` | Streamlit UI、Session State 與流程編排 |
| `llm_service.py` | Prompt、Structured Outputs 與偏好更新 |
| `preference_state.py` | Pydantic schema 與 State 清理 |
| `web_enrichment.py` | 受控 Web query expansion、來源擷取與語意組合 |
| `result_controls.py` | 換一位、換一批與主播向量相似度計算 |
| `literal_search.py` | 姓名、帳號與 PFID 逐字搜尋 |
| `search_index.py` | 倒排索引與標籤計分 |
| `vector_search.py` | 主播 Embedding 與 cosine similarity |
| `recommender.py` | CSV 驗證、reason 對應、候選合併與排名 |
| `result_summary.py` | Grounded LLM 摘要與規則式 fallback |
| `streamer_cards.py` | 聊天推薦與二選一共用的 Top 5 主播卡片 |
| `pairwise.py` | 二選一弱偏好更新、Jaccard 配對與排序 |
| `pages/1_Pairwise_Explorer.py` | 二選一實驗分頁 UI |
| `prepare_data.py` | 本機離線資料正規化工具 |

## 執行方式

建議使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

設定 API Key：

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

或建立 `.streamlit/secrets.toml`：

```toml
OPENAI_API_KEY = "your-api-key"
```

啟動：

```powershell
streamlit run app.py
```

開啟頁面後上傳 `anchors_100.csv`，等待記憶體 Embedding 索引完成，再開始輸入推薦需求。

## API 呼叫與效能

首次上傳每份不同 CSV 時，系統會以 batch 建立全部主播 Embedding，同一 Session 內重複使用。

每輪有效查詢：

1. LLM 更新 Preference State。
2. 若命中外部實體，額外呼叫一次 Web Search；一般查詢不呼叫。
3. Embedding API 建立查詢向量。
4. NumPy 在本機計算 cosine similarity 並完成排名。
5. Top 5 卡片先顯示。
6. 額外呼叫一次 LLM 產生推薦摘要。

二選一頁的本機配對與排名不需要 API；每完成五輪才呼叫一次 LLM 更新摘要，沒有 API Key 或呼叫失敗時使用規則式 fallback。

聊天結果控制使用已快取的召回分數與主播向量在本機重排，不重新計算主播或 query Embedding；只有更新後的推薦摘要會再呼叫一次 LLM。

131 筆資料的矩陣運算成本很低，主要延遲來自網路 API。新的 Session、Server 重啟或不同 CSV 會重新建立主播 Embedding。

## 設計取捨與已知限制

- 131 筆資料使用記憶體索引，不使用向量資料庫或分散式架構。
- 欄位權重與純語意 `0.20` 門檻是 heuristic，尚未以人工 relevance labels 校準。
- 單一標籤可能讓多位主播同為 100%；Embedding 只打破同分，不代表能力高低。
- 姓名與帳號不支援錯字修正與同名消歧。
- LLM 可能誤解多輪修改；標籤清理只能驗證值是否存在，不能驗證語意理解是否正確。
- 缺少觀看、點擊、活躍度與互動行為，因此不是協同過濾或個人化 learning-to-rank。
- 二選一分數只反映少量當前選擇，尚未經真實使用者行為校準或持久化。
- `換一位／換一批` 只在本次查詢維護略過清單，不會跨 Session 學習長期偏好。
- Web query expansion 的觸發由 LLM 判斷，可能漏掉新實體或誤將普通名詞視為實體；搜尋結果也可能隨時間改變。
- 相似主播目前只使用單一主播向量作為基準，尚未加入多主播混合或多樣性約束。

## 資料與隱私

本版本只在使用者明確提到外部作品、角色、IP、遊戲或流行語時，動態使用 OpenAI Responses API 的 Web Search；實際來源會顯示在 Streamlit 查詢結果旁，僅用於 query expansion，不保存成主播資料。`anchors_100.csv` 與 `data_prepared/` 已列入 `.gitignore`。

上傳檔不會由程式寫入專案目錄，但主播文字會送往 OpenAI Embedding API，使用者訊息與摘要 context 也會送往 OpenAI Responses API。正式部署前需確認資料政策；若資料不可傳送第三方，可改用本機 multilingual embedding 與本機 LLM。

## AI 協作說明

AI 工具用於需求拆解、程式實作、重構、Prompt 與 README 草擬。產出透過以下方式驗證：

- Python 語法與 imports 檢查。
- 使用實際 CSV 驗證姓名、PFID、標籤分數與 `reasons` 對應。
- 使用合成案例確認 `tag_score` 優先，`vector_score` 只打破同分。
- Structured Outputs 限制 LLM 回傳 schema。
- `sanitize_preferences()` 移除資料中不存在的模型標籤。
- 摘要只接收 Top 5 metadata 與證據，推薦候選不由 LLM 決定。

## 未來方向

- 姓名與帳號加入高門檻 fuzzy matching 及使用者確認。
- 為 Web query expansion 建立固定實體測試集，評估觸發 precision、來源品質與推薦排序變化。
- 將二選一結果與聊天 Preference State 整合，並加入持久化 Feedback。
- 建立人工 relevance set，以 Precision@K、NDCG@K 校準權重與語意門檻。
- 資料規模擴大後再考慮持久化 Embedding、向量索引與摘要 cache。
