"""Controlled web query expansion for externally named concepts."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str


@dataclass(frozen=True)
class WebEnrichment:
    lookup_query: str
    entity: str
    expanded_query: str
    explanation: str
    sources: tuple[WebSource, ...]


class _WebExpansionOutput(BaseModel):
    entity: str
    expanded_query: str
    explanation: str


def _extract_sources(response: object, limit: int = 3) -> tuple[WebSource, ...]:
    """Extract safe, deduplicated sources returned by the hosted search tool."""
    payload = response.model_dump()  # type: ignore[attr-defined]
    seen: set[str] = set()
    sources: list[WebSource] = []
    for item in payload.get("output", []):
        action = item.get("action") or {}
        for source in action.get("sources") or []:
            url = str(source.get("url") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if url in seen:
                continue
            seen.add(url)
            title = str(source.get("title") or parsed.netloc).strip()
            sources.append(WebSource(title=title, url=url))
            if len(sources) >= limit:
                return tuple(sources)
    return tuple(sources)


def expand_web_query(
    client: OpenAI,
    model: str,
    lookup_query: str,
) -> WebEnrichment:
    """Search one external entity and translate it into recommendation traits."""
    query = " ".join(lookup_query.split())
    if not query:
        raise ValueError("Web lookup query 不可為空")

    response = client.responses.parse(
        model=model,
        instructions="""你負責替主播推薦系統理解外部作品、角色、IP、遊戲或流行語。

請先使用 Web Search 查證輸入的實體，再輸出適合與主播 metadata 比較的繁體中文特徵。
- expanded_query 只保留人格、互動方式、內容主題、才藝、視覺風格或整體氛圍等推薦特徵，使用頓號分隔，最多 80 字。
- 不可聲稱任何主播具備這些特徵，也不可產生主播姓名、帳號或 PFID。
- 不確定輸入指向哪個實體，或來源不足以支持特徵時，expanded_query 留空並在 explanation 簡短說明。
- explanation 最多一句，說明外部資訊如何協助理解，不要放網址或 Markdown。
""",
        input=query,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        text_format=_WebExpansionOutput,
        reasoning={"effort": "low"},
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("Web Search 未回傳可解析的查詢擴充")
    return WebEnrichment(
        lookup_query=query,
        entity=parsed.entity.strip(),
        expanded_query=parsed.expanded_query.strip(),
        explanation=parsed.explanation.strip(),
        sources=_extract_sources(response),
    )


def combine_semantic_query(
    semantic_query: str,
    enrichment: WebEnrichment | None,
) -> str:
    """Keep external knowledge separate while including it in vector retrieval."""
    original = semantic_query.strip()
    expanded = enrichment.expanded_query.strip() if enrichment else ""
    if original and expanded:
        return f"{original}；外部概念特徵：{expanded}"
    return original or expanded
