"""
API를 쓰지 않는 핵심 로직 테스트.  실행: python -m pytest tests -q

코드 리뷰(2026-10-07)에서 지적된 문제들이 다시 생기지 않는지 확인합니다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessageChunk, ToolMessage  # noqa: E402

from agent import EMPTY_ANSWER, clean_markdown, friendly_error, stream_answer  # noqa: E402
from errors import classify, is_quota_or_unavailable  # noqa: E402
from retrieval import KeywordIndex  # noqa: E402


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


# --- stream_answer: 빈 답변, 서두와 답변 구분, 키워드 검색 표시 유지 ---
class FakeAgent:
    def __init__(self, events):
        self.events = events

    def stream(self, _inputs, stream_mode):
        yield from self.events


MODEL = {"langgraph_node": "model"}
TOOLS = {"langgraph_node": "tools"}


def tool_msg(pages, method):
    return ToolMessage(content="...", tool_call_id="1", artifact={"pages": pages, "method": method})


def test_whitespace_only_answer_becomes_fallback():
    out = "".join(stream_answer(FakeAgent([(AIMessageChunk(content="\n "), MODEL)]), [], {}))
    assert out.endswith(EMPTY_ANSWER)


def test_preamble_and_answer_are_separated():
    agent = FakeAgent([
        (AIMessageChunk(content="검색해 보겠습니다."), MODEL),
        (tool_msg([18], "vector"), TOOLS),
        (AIMessageChunk(content="공기압은 33psi입니다."), MODEL),
    ])
    out = "".join(stream_answer(agent, [], {}))
    assert "검색해 보겠습니다.\n\n공기압은" in out


def test_keyword_method_is_kept_across_multiple_searches():
    sources = {}
    agent = FakeAgent([(tool_msg([23], "keyword"), TOOLS), (tool_msg([24], "vector"), TOOLS),
                       (AIMessageChunk(content="답"), MODEL)])
    "".join(stream_answer(agent, [], sources))
    assert sources == {"pages": [23, 24], "method": "keyword"}


def test_clean_markdown_removes_br():
    assert clean_markdown("a<br>b<BR/>c") == "a · b · c"
