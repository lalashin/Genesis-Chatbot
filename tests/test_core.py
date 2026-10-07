"""
API를 쓰지 않는 핵심 로직 테스트.  실행: python -m pytest tests -q

코드 리뷰(2026-10-07)에서 지적된 문제들이 다시 생기지 않는지 확인합니다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessageChunk  # noqa: E402

from agent import (  # noqa: E402
    EMPTY_ANSWER, Assistant, build_prompt, clean_markdown, friendly_error, search_query, sources_of,
)
from errors import classify, is_quota_or_unavailable  # noqa: E402
from retrieval import KeywordIndex, SearchResult  # noqa: E402


# --- errors.classify: 숫자가 섞인 메시지를 한도 오류로 오분류하지 않는다 ---
def test_classify_quota_and_ignores_embedded_numbers():
    assert classify(Exception("429 RESOURCE_EXHAUSTED ... PerMinute")) == "quota_minute"
    assert classify(Exception("RESOURCE_EXHAUSTED EmbedContentRequestsPerDay")) == "quota_day"
    assert classify(Exception("InvalidArgument: prompt has 14290 tokens, id 4041")) is None
    assert not is_quota_or_unavailable(Exception("value 15030 out of range"))
    assert classify(Exception("503 UNAVAILABLE")) == "unavailable"


def test_friendly_error_is_korean_for_quota():
    assert "30초" in friendly_error(Exception("429 RESOURCE_EXHAUSTED"))


# --- KeywordIndex: 한 글자 질의, 빈 문서 ---
def test_keyword_single_char_query():
    idx = KeywordIndex([Document(page_content="휠 볼트 체결 토크"), Document(page_content="와이퍼 교체")])
    assert idx.search("휠", k=1)[0].page_content.startswith("휠")


def test_keyword_empty_docs_no_zero_division():
    idx = KeywordIndex([Document(page_content=""), Document(page_content="")])
    assert idx.search("타이어", k=3) == []


def test_keyword_bigram_finds_korean_without_spaces():
    idx = KeywordIndex([Document(page_content="타이어 공기압 경보 시스템"), Document(page_content="엔진 오일")])
    assert idx.search("타이어공기압경고등", k=1)[0].page_content.startswith("타이어")


# --- Assistant (RAG 체인): 이어지는 질문 검색어, 빈 답변, 프롬프트 구성 ---
class FakeModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.prompt = None

    def stream(self, prompt):
        self.prompt = prompt
        for c in self.chunks:
            yield AIMessageChunk(content=c)


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_follow_up_question_is_combined_with_previous():
    m = msgs(("user", "엔진 오일 용량은?"), ("assistant", "6.2ℓ"), ("user", "그럼 냉각수는?"))
    assert search_query(m) == "엔진 오일 용량은? 그럼 냉각수는?"


def test_new_topic_question_is_searched_alone():
    m = msgs(("user", "엔진 오일 용량은?"), ("assistant", "6.2ℓ"), ("user", "스마트 크루즈 컨트롤 사용법 알려줘"))
    assert search_query(m) == "스마트 크루즈 컨트롤 사용법 알려줘"


def test_whitespace_only_answer_becomes_fallback():
    a = Assistant(retriever=None, model=FakeModel(["\n ", " "]))
    out = "".join(a.stream(msgs(("user", "질문")), SearchResult([], "vector")))
    assert out.endswith(EMPTY_ANSWER)


def test_prompt_contains_manual_excerpt_and_recent_history_only():
    docs = [Document(page_content="2.5 터보 6.2 ℓ", metadata={"page": 22})]
    history = [{"role": "assistant", "content": "안녕하세요", "greeting": True}]
    for i in range(6):
        history += msgs(("user", f"질문{i}"), ("assistant", f"답{i}"))
    history += msgs(("user", "현재 질문"))
    prompt = build_prompt(history, SearchResult(docs, "vector"))
    assert "[23페이지]" in prompt[0].content and "6.2" in prompt[0].content
    contents = [p.content for p in prompt[1:]]
    assert contents[-1] == "현재 질문" and "안녕하세요" not in contents
    assert len(contents) == 7  # 최근 3턴(6개) + 현재 질문


def test_sources_of_lists_pages_and_method():
    docs = [Document(page_content="a", metadata={"page": 22}), Document(page_content="b", metadata={"page": 713})]
    assert sources_of(SearchResult(docs, "keyword")) == {"pages": [23, 714], "method": "keyword"}


def test_clean_markdown_removes_br():
    assert clean_markdown("a<br>b<BR/>c") == "a · b · c"
