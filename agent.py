"""
매뉴얼 Q&A 에이전트.

- search_manual 도구가 검색한 매뉴얼 페이지를 출처(sources)로 함께 돌려줍니다.
- stream_answer()는 답변을 토큰 단위로 흘려보내 화면에 바로 표시할 수 있게 합니다.
- friendly_error()는 API 오류를 시연 중에도 당황하지 않을 한국어 안내로 바꿉니다.
"""
import re
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from errors import classify
from retrieval import ManualRetriever, page_label
from settings import LLM_MODEL, LLM_TEMPERATURE, SEARCH_K, SYSTEM_PROMPT

EMPTY_ANSWER = "답변을 생성하지 못했습니다. 질문을 조금 바꿔서 다시 물어봐 주세요."


def build_agent(retriever: ManualRetriever):
    @tool(response_format="content_and_artifact")
    def search_manual(query: str):
        """제네시스 차량 매뉴얼을 검색합니다. 차량 문제, 기능 사용법, 유지보수 정보 등을 찾을 때 사용하세요."""
        result = retriever.search(query, k=SEARCH_K)
        pages = sorted({p for p in (page_label(d) for d in result.docs) if p})
        if not result.docs:
            return "관련 정보를 찾을 수 없습니다.", {"pages": [], "method": result.method}
        content = "\n\n".join(f"[{page_label(d)}페이지]\n{d.page_content}" for d in result.docs)
        return content, {"pages": pages, "method": result.method}

    model = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=1)
    return create_agent(model, [search_manual], system_prompt=SYSTEM_PROMPT)


def to_langchain_messages(messages: list[dict]) -> list:
    history = []
    for m in messages:
        if m.get("greeting"):
            continue
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history


def stream_answer(agent, messages: list[dict], sources: dict) -> Iterator[str]:
    """답변 텍스트를 조각 단위로 yield 합니다. 검색한 페이지는 sources에 모읍니다.

    sources = {"pages": [...], "method": "vector" | "keyword"}
    """
    produced = False      # 공백이 아닌 글자를 한 번이라도 냈는지 (빈 답변이 기록에 남으면 다음 요청이 400)
    need_break = False    # 도구 호출 전 서두("검색해 볼게요")와 최종 답변이 붙지 않도록 줄바꿈
    stream = agent.stream({"messages": to_langchain_messages(messages)}, stream_mode="messages")
    for chunk, meta in stream:
        if isinstance(chunk, ToolMessage) and isinstance(chunk.artifact, dict):
            sources["pages"] = sorted(set(sources.get("pages", [])) | set(chunk.artifact["pages"]))
            # 여러 번 검색했으면 한 번이라도 키워드 검색을 쓴 경우 keyword로 표시
            if sources.get("method") != "keyword":
                sources["method"] = chunk.artifact["method"]
            need_break = produced
        elif isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model":
            text = chunk.text
            if not text:
                continue
            if need_break:
                yield "\n\n"
                need_break = False
            produced = produced or bool(text.strip())
            yield text
    if not produced:
        yield EMPTY_ANSWER


def clean_markdown(text: str) -> str:
    """모델이 표 안에 넣는 <br> 같은 HTML 태그는 st.markdown에서 그대로 보이므로 정리합니다."""
    return re.sub(r"<br\s*/?>", " · ", text, flags=re.IGNORECASE)


FRIENDLY_MESSAGES = {
    "quota_day": "오늘 사용할 수 있는 무료 AI 사용량을 모두 썼어요. 내일 다시 이용해 주세요.",
    "quota_minute": "질문이 몰려 잠시 쉬어가는 중이에요. 30초쯤 뒤에 다시 질문해 주세요.",
    "unavailable": "AI 서버가 잠시 혼잡해요. 잠시 후 다시 시도해 주세요.",
    "not_found": "AI 모델을 찾을 수 없어요. 관리자에게 모델 설정(settings.py) 확인을 요청해 주세요.",
    "auth": "AI 서비스 인증에 실패했어요. API 키 설정을 확인해 주세요.",
}


def friendly_error(e: Exception) -> str:
    return FRIENDLY_MESSAGES.get(classify(e), f"답변 중 문제가 발생했어요. 다시 시도해 주세요. ({type(e).__name__})")
