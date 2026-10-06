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

    def fake_stream_answer(_agent, messages, sources):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(RATE_LIMIT)
        sources.update({"pages": [23], "method": "vector"})
        yield "엔진 오일은 6.2ℓ입니다."

    monkeypatch.setattr(agent, "stream_answer", fake_stream_answer)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="chat_toggle").click().run()
    return at


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


def test_daily_quota_notice(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def fake(_agent, _messages, _sources):
        raise Exception("429 RESOURCE_EXHAUSTED 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
        yield  # noqa: unreachable — 제너레이터로 만들기 위함

    monkeypatch.setattr(agent, "stream_answer", fake)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="chat_toggle").click().run()
    at.chat_input[0].set_value("질문").run()
    assert not at.exception
    assert any("내일" in e.value for e in at.error)
