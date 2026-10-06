"""
GENESIS AI Assistant - 제네시스 차량 매뉴얼 RAG 챗봇 (화면)

구성
- settings.py           : 모델 이름, 프롬프트 등 설정
- retrieval.py          : 매뉴얼 검색 (벡터 검색 + 한도 초과 시 키워드 검색)
- agent.py              : 답변 에이전트, 스트리밍, 오류 안내
- voice.py              : 음성 받아쓰기 (Gemini)
- styles.css            : 배경 이미지 등 테마로 못 하는 스타일
- .streamlit/config.toml: 색상·폰트 테마
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import build_agent, clean_markdown, friendly_error, stream_answer
from retrieval import ManualRetriever
from settings import VOICE_AUTO_SEND
from vectorstore_config import PERSIST_DIR
from voice import transcribe

st.set_page_config(page_title="제네시스 매뉴얼 챗봇", page_icon=":material/directions_car:")
st.html(Path(__file__).with_name("styles.css"))

GREETING = {"role": "assistant", "content": "안녕하세요! 제네시스 차량에 대해 궁금한 점을 물어보세요.", "greeting": True}
SUGGESTIONS = {
    ":material/tire_repair: 타이어 공기압": "타이어 공기압은 얼마나 넣어야 해?",
    ":material/key: 스마트 키 배터리": "스마트 키 배터리가 방전되면 시동은 어떻게 걸어?",
    ":material/warning: 엔진 과열": "엔진이 과열되면 어떻게 해야 해?",
    ":material/local_gas_station: 주유구 열기": "주유구는 어떻게 열어?",
}

# === 1. API 키 (Streamlit Secrets 우선, 없으면 로컬 .env) ===
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # 로컬에는 secrets.toml이 없을 수 있음
if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 .env 파일을 확인해주세요.")
    st.stop()


# === 2. 리소스 (한 번만 생성) ===
@st.cache_resource
def load_agent():
    # 벡터 DB는 build_vectorstore.py로 미리 만들어 둡니다 (docs/decisions/002-prebuilt-vectorstore.md)
    if not os.path.exists(PERSIST_DIR):
        st.error("벡터 DB가 없습니다. `python build_vectorstore.py`를 먼저 실행해주세요.")
        st.stop()
    return build_agent(ManualRetriever())


agent = load_agent()

# === 3. 세션 상태 ===
st.session_state.setdefault("messages", [GREETING])
st.session_state.setdefault("show_chat", False)
st.session_state.setdefault("voice_enabled", True)
st.session_state.setdefault("voice_notice", None)


def toggle_chat():
    st.session_state.show_chat = not st.session_state.show_chat


def clear_chat():
    st.session_state.messages = [GREETING]


# === 4. 화면 상단 ===
st.toggle(
    "음성으로 질문하기" if not st.session_state.voice_enabled else "음성 질문 사용 중 (입력창의 마이크 버튼)",
    key="voice_enabled",
)
st.title("GENESIS AI Assistant")

with st.sidebar:
    st.title("GENESIS Assistant")
    guide_tab, manage_tab = st.tabs([":material/lightbulb: 가이드", ":material/settings: 대화 관리"])
    with guide_tab:
        st.subheader("사용법")
        st.markdown(
            """
1. **우측 하단 버튼**을 눌러 대화를 시작하세요.
2. **차량 기능, 유지보수, 문제 해결**에 대해 물어보세요.
3. **음성 질문**: 입력창의 :material/mic: 버튼으로 녹음 → 받아쓴 문장을 확인하고 전송하세요.
4. 답변 아래 **참고한 매뉴얼 페이지**가 표시됩니다.
"""
        )
    with manage_tab:
        st.button("대화 내용 지우기", icon=":material/delete:", on_click=clear_chat, width="stretch")
        st.toggle("받아쓴 문장 바로 전송", value=VOICE_AUTO_SEND, key="voice_auto_send",
                  help="끄면 받아쓴 문장을 입력창에 넣어 확인 후 전송합니다.")

st.button(
    "",
    icon=":material/close:" if st.session_state.show_chat else ":material/chat:",
    key="chat_toggle",
    on_click=toggle_chat,
    help="대화창 닫기" if st.session_state.show_chat else "대화 시작",
)


def render_sources(sources: dict | None):
    if not sources or not sources.get("pages"):
        return
    pages = ", ".join(f"{p}쪽" for p in sources["pages"])
    note = " · 키워드 검색" if sources.get("method") == "keyword" else ""
    st.caption(f":material/menu_book: 참고한 매뉴얼: {pages}{note}")


# === 5. 대기 화면 ===
if not st.session_state.show_chat:
    st.html(
        "<div class='welcome'><h2>GENESIS AI</h2><p>우측 하단 버튼을 눌러 대화를 시작하세요</p></div>"
    )
    st.stop()

# === 6. 대화 화면 ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

def pick_suggestion():
    # 고른 질문을 꺼내고 선택은 바로 지웁니다. 선택이 남아 있으면 답변이 실패했을 때
    # 이후 아무 rerun에서나 같은 질문이 반복 전송됩니다.
    picked = st.session_state.suggestion
    st.session_state.suggestion = None
    if picked:
        st.session_state.pending_prompt = SUGGESTIONS[picked]


prompt = st.session_state.pop("pending_prompt", None)
if len(st.session_state.messages) == 1:  # 첫 질문 전에만 예시 질문 표시
    st.pills("예시 질문", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion",
             on_change=pick_suggestion)

if st.session_state.voice_notice:
    st.info(st.session_state.voice_notice, icon=":material/mic:")
    st.session_state.voice_notice = None

# 받아쓴 문장은 입력창을 만들기 전에 넣어야 합니다 (만든 뒤에는 값을 바꿀 수 없음)
if "pending_input" in st.session_state:
    st.session_state.chat_input = st.session_state.pop("pending_input")

submission = st.chat_input(
    "질문을 입력하세요 (예: 타이어 공기압은?)",
    key="chat_input",
    accept_audio=st.session_state.voice_enabled,
    submit_mode="disable",
)

if submission:
    text = submission if isinstance(submission, str) else (submission.text or "")
    audio = None if isinstance(submission, str) else getattr(submission, "audio", None)
    if text.strip():
        prompt = text.strip()
    elif audio is not None:
        # 음성 질문: Gemini로 받아쓰기 (차량 용어 힌트 포함)
        try:
            with st.spinner("음성을 받아쓰는 중..."):
                heard = transcribe(audio.getvalue(), audio.type or "audio/wav")
        except Exception as e:
            st.error(friendly_error(e))
            st.stop()
        if not heard:
            st.warning("음성을 알아듣지 못했어요. 조금 더 크게, 또렷하게 다시 말해 주세요.", icon=":material/mic_off:")
        elif st.session_state.voice_auto_send:
            prompt = heard
        else:
            # 받아쓴 문장을 입력창에 넣어 확인 후 전송 (시연 중 오인식을 바로 고칠 수 있게)
            st.session_state.pending_input = heard
            st.session_state.voice_notice = f"받아쓴 문장을 입력창에 넣었어요. 확인 후 전송하세요: **{heard}**"
            st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources: dict = {}
        try:
            with st.spinner("매뉴얼을 찾는 중..."):
                answer = st.write_stream(stream_answer(agent, st.session_state.messages, sources))
        except Exception as e:
            st.session_state.messages.pop()  # 실패한 질문은 대화 기록에서 제외
            st.error(friendly_error(e), icon=":material/error:")
        else:
            answer = clean_markdown(answer if isinstance(answer, str) else "".join(map(str, answer)))
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()
