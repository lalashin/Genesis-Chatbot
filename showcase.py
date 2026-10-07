"""
'개발 과정 한눈에 보기' 팝업 (데모·교육용).

앱 제목 아래 작은 링크로 열며, 세 가지를 보여 줍니다.
- 작동 원리: 질문이 답변이 되기까지 (순서도)
- 개발 여정: 단계별로 겪은 문제와 해결
- 개선 효과: 측정한 전후 수치 (숫자 타일)

수치는 docs/eval/ 의 측정 결과에서 가져왔습니다. 다시 측정하면 여기 값도 갱신하세요.
"""
import streamlit as st

REPO_DOCS = "https://github.com/lalashin/Genesis-Chatbot/tree/master/docs"

HOW_IT_WORKS = """
flowchart LR
    Q1["⌨️ 텍스트 질문"] --> S
    Q2["🎙️ 음성 질문"] --> T["Gemini 받아쓰기<br/>(차량 용어 힌트)"] --> S
    S["🔎 매뉴얼 검색<br/>질문과 비슷한 내용 3곳"] --> A["🤖 Gemini 답변 작성<br/>(검색한 내용만 근거로)"]
    A --> R["💬 답변 + 📖 참고 페이지"]
    S -. 무료 한도 초과 시 .-> K["키워드 검색으로 대체"] -.-> A
"""

DATA_PREP = """
flowchart LR
    P["📕 매뉴얼 PDF<br/>767쪽"] --> C["✂️ 1,000자씩 나누기<br/>1,005조각"] --> E["🔢 Gemini 임베딩<br/>조각마다 숫자 768개"] --> D["🗄️ 벡터 DB<br/>(미리 만들어 저장)"]
"""

JOURNEY = """
flowchart LR
    P1["1부 · 바이브 코딩<br/>01~07<br/>AI 채팅만으로 첫 버전"] --> P2["2부 · 점검과 전환<br/>08~09<br/>OpenAI → Gemini, 버그 수정"] --> P3["3부 · 데모 고도화<br/>10~11<br/>근거·평가·속도"]
"""

JOURNEY_TABLE = """
| 단계 | 한 일 | 겪은 문제 → 해결 |
|---|---|---|
| 01~02 | PDF 검색 챗봇, Streamlit 배포 | 매뉴얼이 문단 단위로 안 나뉘던 줄바꿈 기호 오류 → 나중에 코드 점검으로 발견 |
| 03~07 | 대화 기억, 음성 질문, 모바일 대응 | 음성 화면이 안 닫힘 → 화면 갱신 때 사라지는 코드 구조가 원인 |
| 08 | OpenAI → Google Gemini | 모델 404·임베딩 한도 429 → 모델 실제 호출로 확인, 벡터 DB 미리 만들기 |
| 09 | 파일 정리, 음성 버그 수정 | 한 문장이 10번 전송 → 듣기 1회당 1번만 전송 |
| 10 | 데모 고도화 | "차체→자체" 오인식 → 차량 용어 힌트 받아쓰기 / 출처 표시 / 평가 세트 |
| 11 | 답변 속도 개선 | 모델을 2~3번 호출 → 검색 먼저, 모델 1번 호출 |
"""


@st.dialog("개발 과정 한눈에 보기", width="large", icon=":material/account_tree:")
def show_story():
    how, journey, impact = st.tabs([
        ":material/route: 작동 원리",
        ":material/history: 개발 여정",
        ":material/trending_up: 개선 효과",
    ])

    with how:
        st.markdown("**질문이 답변이 되기까지**")
        st.mermaid_chart(HOW_IT_WORKS)
        st.markdown("**미리 해 두는 준비 (한 번만)**")
        st.mermaid_chart(DATA_PREP)
        st.caption("매뉴얼을 미리 숫자(벡터)로 바꿔 두면, 질문이 들어올 때 뜻이 비슷한 부분을 바로 찾을 수 있습니다.")

    with journey:
        st.mermaid_chart(JOURNEY)
        st.markdown(JOURNEY_TABLE)
        st.caption("실패한 시도와 오류도 모두 기록해 두었습니다. 자세한 과정은 아래 링크의 개발 일지에서 볼 수 있습니다.")

    with impact:
        st.caption("같은 조건에서 바꾸기 전과 후를 측정한 값입니다.")
        row1 = st.container(horizontal=True)
        with row1:
            st.metric("음성 받아쓰기 정확도", "10 / 10", "+4 (힌트 없을 때 6 / 10)", border=True,
                      help="차량 용어가 들어간 질문 10개를 합성 음성으로 측정. '차체→자체' 같은 오인식이 사라짐")
            st.metric("매뉴얼 검색 정확도", "100%", "+10%p (키워드 검색 90%)", border=True,
                      help="질문 20개 중 정답 페이지가 검색 상위 3개 안에 든 비율 (Hit@3)")
        row2 = st.container(horizontal=True)
        with row2:
            st.metric("첫 글자가 보이기까지", "1.8초", "-1.3초 (개선 전 3.1초)", delta_color="inverse", border=True,
                      help="같은 질문 4개 평균. 검색을 먼저 하고 AI를 한 번만 부르도록 구조 변경")
            st.metric("답변이 끝나기까지", "3.9초", "-2.2초 (개선 전 6.1초)", delta_color="inverse", border=True,
                      help="답변을 핵심 위주로 짧게 쓰도록 조정")
            st.metric("질문 1개당 AI 호출", "1회", "-1–2회 (개선 전 2–3회)", delta_color="inverse", border=True,
                      help="호출이 줄어 무료 사용 한도도 2~3배 여유가 생김")
        st.caption("측정 기록: docs/eval/ · 로컬 PC 기준이며 배포 서버에서는 네트워크에 따라 조금 더 걸릴 수 있습니다.")

    st.link_button("전체 개발 기록 보기 (GitHub)", REPO_DOCS, icon=":material/open_in_new:", type="tertiary")
