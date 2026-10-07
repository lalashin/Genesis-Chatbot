"""
매뉴얼 Q&A 어시스턴트 (RAG 체인).

예전에는 LangChain Agent가 "검색할지"를 모델에게 먼저 물어본 뒤 검색하고 다시 답변을 만들어
질문 1개에 모델을 2~3번 호출했습니다(첫 글자까지 2~4초). 매뉴얼 Q&A는 거의 항상 검색이 필요하므로
**검색을 먼저 하고 모델은 1번만 호출**하도록 바꿨습니다 → docs/decisions/007-rag-chain-for-speed.md

- search(): 질문(이어지는 질문이면 직전 질문과 합쳐서)으로 매뉴얼 검색
- stream(): 검색 결과를 근거로 답변을 토큰 단위로 생성
- friendly_error(): API 오류를 한국어 안내로 변환
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from errors import classify
from retrieval import ManualRetriever, SearchResult, page_label
from settings import (
    HISTORY_TURNS,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_THINKING_LEVEL,
    SEARCH_K,
    SYSTEM_PROMPT,
)

EMPTY_ANSWER = "답변을 생성하지 못했습니다. 질문을 조금 바꿔서 다시 물어봐 주세요."

# "그럼 냉각수는?"처럼 앞 질문에 기대는 짧은 질문은 직전 질문과 합쳐서 검색합니다.
FOLLOW_UP_WORDS = ("그럼", "그러면", "그건", "그거", "이건", "이거", "저건", "거기", "그때", "그 다음", "아까", "방금")
FOLLOW_UP_MAX_LEN = 12


@dataclass
class Assistant:
    retriever: ManualRetriever
    model: ChatGoogleGenerativeAI

    def search(self, messages: list[dict]) -> SearchResult:
        return self.retriever.search(search_query(messages), k=SEARCH_K)

    def stream(self, messages: list[dict], result: SearchResult) -> Iterator[str]:
        """검색 결과를 근거로 답변을 조각 단위로 yield 합니다. 공백뿐이면 대체 문구."""
        produced = False
        for chunk in self.model.stream(build_prompt(messages, result)):
            text = chunk.text
            if not text:
                continue
            produced = produced or bool(text.strip())
            yield text
        if not produced:
            yield EMPTY_ANSWER


def build_assistant(retriever: ManualRetriever) -> Assistant:
    model = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        thinking_config={"thinking_level": LLM_THINKING_LEVEL},
        max_retries=1,
    )
    return Assistant(retriever, model)


def user_questions(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def search_query(messages: list[dict]) -> str:
    questions = user_questions(messages)
    current = questions[-1]
    if len(questions) >= 2 and (
        len(current.replace(" ", "")) <= FOLLOW_UP_MAX_LEN or current.startswith(FOLLOW_UP_WORDS)
    ):
        return f"{questions[-2]} {current}"
    return current


def format_context(result: SearchResult) -> str:
    if not result.docs:
        return "(관련 매뉴얼 내용을 찾지 못했습니다)"
    return "\n\n".join(f"[{page_label(d)}페이지]\n{d.page_content}" for d in result.docs)


def build_prompt(messages: list[dict], result: SearchResult) -> list:
    """시스템 프롬프트 + 매뉴얼 발췌 + 최근 대화 + 현재 질문."""
    system = f"{SYSTEM_PROMPT}\n\n## 매뉴얼 발췌 (이 내용만 근거로 답변)\n{format_context(result)}"
    history = [m for m in messages if not m.get("greeting")]
    recent = history[-(HISTORY_TURNS * 2 + 1):]  # 최근 N턴 + 현재 질문
    return [SystemMessage(system)] + [
        (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]) for m in recent
    ]


def sources_of(result: SearchResult) -> dict:
    pages = sorted({p for p in (page_label(d) for d in result.docs) if p})
    return {"pages": pages, "method": result.method}


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
