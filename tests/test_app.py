"""
앱 화면 단위 테스트 (Streamlit AppTest). API를 호출하지 않습니다.  실행: python -m pytest tests -q

성공 기준 S4 "429 발생 시 한국어 안내, 앱이 멈추지 않음"을 앱 화면에서 확인합니다 (갭 분석 G4).
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

import agent  # noqa: E402

APP = os.path.join(ROOT, "streamlit_app.py")
RATE_LIMIT = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'"
)


@pytest.fixture
def app(monkeypatch):
    """첫 질문은 답변 모델 429로 실패, 그다음 질문은 정상 답변하도록 stream_answer를 바꿔 끼운 앱."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # 실제 호출은 하지 않음
    calls = {"n": 0}

    def fake_stream(self, messages, result):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(RATE_LIMIT)
        yield "엔진 오일은 6.2ℓ입니다."

    fake_search_results(monkeypatch)
    monkeypatch.setattr(agent.Assistant, "stream", fake_stream)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="chat_toggle").click().run()
    return at


def fake_search_results(monkeypatch):
    from langchain_core.documents import Document

    from retrieval import SearchResult

    doc = Document(page_content="2.5 터보 6.2 ℓ", metadata={"page": 22})
    monkeypatch.setattr(agent.Assistant, "search", lambda self, messages: SearchResult([doc], "vector"))


def user_messages(at):
    return [m["content"] for m in at.session_state["messages"] if m["role"] == "user"]


def test_rate_limit_shows_korean_notice_and_app_recovers(app):
    # 1) 답변 모델 429: 한국어 안내, 실패한 질문은 기록에서 제외, 앱 예외 없음
    app.chat_input[0].set_value("엔진 오일 용량은?").run()
    assert not app.exception
    assert any("30초" in e.value for e in app.error), [e.value for e in app.error]
    assert user_messages(app) == []

    # 2) 화면이 다시 그려져도 실패한 질문이 재전송되지 않음
    app.toggle(key="voice_enabled").set_value(False).run()
    assert user_messages(app) == []
    assert not app.error

    # 3) 다음 질문은 정상 처리 (앱이 멈추지 않음), 출처도 저장
    app.chat_input[0].set_value("엔진 오일 용량은?").run()
    assert not app.exception
    msgs = app.session_state["messages"]
    assert [m["role"] for m in msgs[-2:]] == ["user", "assistant"]
    assert msgs[-1]["content"] == "엔진 오일은 6.2ℓ입니다."
    assert msgs[-1]["sources"] == {"pages": [23], "method": "vector"}


class FakeAudio:
    type = "audio/wav"

    def getvalue(self):
        return b"RIFF-fake"


class FakeSubmission:
    """녹음만 하고 글자는 없는 st.chat_input 제출 값 (AppTest는 녹음 제출을 지원하지 않아 직접 만듦)."""
    text = ""
    audio = FakeAudio()


def voice_app(monkeypatch, auto_send: bool):
    """입력창이 녹음을 한 번 제출하고, 받아쓰기는 고정 문장을 돌려주도록 바꿔 끼운 앱."""
    import streamlit
    import voice

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(voice, "transcribe", lambda audio, mime: "차체에 흠집이 났을 때 관리 방법")
    fake_search_results(monkeypatch)
    monkeypatch.setattr(agent.Assistant, "stream", lambda *_: iter(["세차 후 왁스를 바르세요."]))

    real_chat_input = streamlit.chat_input
    state = {"submit": False}

    def fake_chat_input(*args, **kwargs):
        real_chat_input(*args, **kwargs)  # 화면에는 진짜 입력창을 그림
        if state["submit"]:
            state["submit"] = False
            return FakeSubmission()
        return None

    monkeypatch.setattr(streamlit, "chat_input", fake_chat_input)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="chat_toggle").click().run()
    at.toggle(key="voice_auto_send").set_value(auto_send).run()
    state["submit"] = True
    return at.run()


def test_voice_auto_send_shows_transcript_as_voice_question(monkeypatch):
    at = voice_app(monkeypatch, auto_send=True)
    assert not at.exception
    user = [m for m in at.session_state["messages"] if m["role"] == "user"]
    assert user == [{"role": "user", "content": "차체에 흠집이 났을 때 관리 방법", "voice": True}]
    assert any("음성 질문" in c.value for c in at.caption)  # 받아쓴 문장임을 표시
    assert at.session_state["messages"][-1]["content"] == "세차 후 왁스를 바르세요."


def test_voice_confirm_mode_puts_transcript_in_input(monkeypatch):
    at = voice_app(monkeypatch, auto_send=False)
    assert not at.exception
    assert [m for m in at.session_state["messages"] if m["role"] == "user"] == []  # 아직 전송 안 됨
    assert at.chat_input[0].value == "차체에 흠집이 났을 때 관리 방법"  # 입력창에 들어가 확인 대기


def test_daily_quota_notice(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def fake(self, _messages, _result):
        raise Exception("429 RESOURCE_EXHAUSTED 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
        yield  # noqa: unreachable — 제너레이터로 만들기 위함

    fake_search_results(monkeypatch)
    monkeypatch.setattr(agent.Assistant, "stream", fake)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="chat_toggle").click().run()
    at.chat_input[0].set_value("질문").run()
    assert not at.exception
    assert any("내일" in e.value for e in at.error)


def test_story_dialog_opens_without_error(monkeypatch):
    """'개발 과정 보기' 링크 → 팝업(순서도·표·수치 타일)이 오류 없이 열림."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="story_link").click().run()
    assert not at.exception
    assert any("음성 받아쓰기 정확도" in m.label for m in at.metric)
