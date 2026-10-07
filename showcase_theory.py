"""
'개발 과정 보기' 팝업의 이론 탭 두 개 (교육용).

- 임베딩 원리: 한 줄 정의, 질문×매뉴얼 유사도 히트맵(실제 데이터), 키워드 vs 임베딩, 직접 해 보기
- AI 설정·페르소나: 실제 시스템 프롬프트(settings.py에서 그대로 읽음), 규칙별 이유, 설정값
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import settings
from retrieval import ManualRetriever, page_label
from vectorstore_config import get_embeddings
from voice import GLOSSARY

MAP_FILE = Path(__file__).with_name("showcase_data") / "embedding_map.json"

PERSONA_RULES = """
| 규칙 | 왜 넣었나 | 없으면 생기는 일 |
|---|---|---|
| 매뉴얼 발췌**만** 근거로 | 지어내기(환각) 방지 | 매뉴얼에 없는 일반 상식을 사실처럼 말함 |
| 없으면 "찾지 못했다"고 말하기 | 모르는 걸 인정하게 | 그럴듯한 오답 |
| 안전 내용 강조 | 차량 매뉴얼의 핵심 가치 | 경고·주의사항이 묻힘 |
| 수치·절차는 **그대로** 제시 | 실제로 쓸모 있는 답 | "모델마다 다르니 라벨을 확인하세요" 같은 일반론 |
| 사양별로 다르면 표·목록 | 2.5 터보 / 3.5 터보처럼 비교 | 숫자가 문장 속에 섞여 헷갈림 |
| 첫 문장에서 바로 답, 머리말 금지 | 핵심 먼저 | "핵심 답:" 같은 머리말을 그대로 출력한 사례가 있었음 |
| 10줄 이내, 반복·인사말 금지 | 답변 속도와 가독성 | 긴 답변 → 끝까지 쓰는 데 3~4초 더 걸림 |
"""

PROMPT_FLOW = """
flowchart LR
    S["🧑‍✈️ 시스템 지시<br/>(페르소나·답변 규칙)"] --> M
    E["📖 매뉴얼 발췌<br/>검색한 조각 3개"] --> M
    H["💬 최근 대화<br/>3턴"] --> M
    Q["❓ 현재 질문"] --> M
    M["Gemini에 한 번에 전달"] --> A["답변"]
"""


@st.cache_data
def load_map() -> dict:
    return json.loads(MAP_FILE.read_text(encoding="utf-8"))


@st.cache_resource
def manual_vectors():
    """매뉴얼 1,005조각의 벡터를 단위 벡터 행렬로 (직접 해 보기에서 코사인 유사도 계산용)."""
    retriever = ManualRetriever()
    raw = retriever.vectorstore.get(include=["embeddings", "documents", "metadatas"])
    X = np.asarray(raw["embeddings"], dtype=float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, raw["documents"], raw["metadatas"], retriever


def render_embedding_tab():
    st.markdown(
        "**임베딩은 문장의 '뜻'을 숫자 768개로 바꾸는 것**입니다. "
        "뜻이 비슷한 문장은 비슷한 숫자가 나오므로, 숫자끼리 가까운 것을 찾으면 "
        "**단어가 달라도 같은 뜻의 매뉴얼**을 찾을 수 있습니다."
    )

    st.markdown("##### 매뉴얼 단어를 피해서 물어도, 같은 주제의 매뉴얼이 가장 가깝습니다")
    data = load_map()
    topics, questions, sim = data["topics"], data["questions"], data["similarity"]
    short = [f"{i + 1}. {q if len(q) <= 10 else q[:9] + '…'}" for i, q in enumerate(questions)]  # 축 이름표는 짧게
    cells = pd.DataFrame([
        {"질문": short[i], "질문 전체": questions[i], "매뉴얼 조각": topics[j], "유사도": sim[i][j]}
        for i in range(len(topics)) for j in range(len(topics))
    ])
    # 크기(유사도)는 한 색상의 밝기로: 낮을수록 어둡게, 높을수록 밝게 (dataviz 순차 팔레트, 다크 배경 #141414 기준 검증 통과)
    color = alt.Color("유사도:Q", title="유사도",
                      scale=alt.Scale(domain=[0.45, 0.8], range=["#1c5cab", "#3987e5", "#6da7ec", "#cde2fb"],
                                      clamp=True))
    heat = alt.Chart(cells).mark_rect(stroke="#141414", strokeWidth=2, cornerRadius=4).encode(
        x=alt.X("매뉴얼 조각:N", sort=topics, title="매뉴얼 조각 (주제)", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("질문:N", sort=short, title=None, axis=alt.Axis(labelLimit=200)),
        color=color,
        tooltip=["질문 전체:N", "매뉴얼 조각:N", alt.Tooltip("유사도:Q", format=".2f")],
    )
    text = alt.Chart(cells).mark_text(fontSize=13).encode(
        x=alt.X("매뉴얼 조각:N", sort=topics), y=alt.Y("질문:N", sort=short),
        text=alt.Text("유사도:Q", format=".2f"),
        color=alt.condition("datum['유사도'] > 0.66", alt.value("#0a0a0a"), alt.value("#e5e5e5")),
    )
    st.altair_chart((heat + text).properties(height=300), width="stretch")
    st.caption("칸의 숫자는 질문과 매뉴얼 조각의 **유사도**(1에 가까울수록 뜻이 비슷함, 768차원 그대로 계산). "
               "**대각선이 각 줄에서 가장 밝으면** 질문이 자기 주제의 매뉴얼을 찾은 것입니다. "
               "예: '바퀴에 바람'에는 '공기압'이라는 단어가 없지만 타이어 공기압 조각과 가장 가깝습니다.  \n"
               "질문 전체: " + " · ".join(f"{i + 1}. {q}" for i, q in enumerate(questions)))

    st.markdown("##### 키워드 검색 vs 임베딩 검색")
    st.markdown(
        "| | 키워드 검색 | 임베딩 검색 (이 챗봇 기본) |\n|---|---|---|\n"
        "| 찾는 방식 | 같은 **글자**가 있는지 | **뜻**이 비슷한지 |\n"
        "| \"주유구 어떻게 열어?\" | 매뉴얼 용어는 \"연료 주입구\"라 놓칠 수 있음 | 같은 뜻으로 찾음 |\n"
        "| 검색 정확도 (질문 20개, Hit@3) | 90% | **100%** |\n"
        "| 비용 | API 없음 (한도 초과 시 대체용) | 질문마다 임베딩 1회 |"
    )

    st.markdown("##### 직접 해 보기")
    with st.form("embed_try", border=False):
        query = st.text_input("질문", placeholder="예: 엔진이 너무 뜨거워졌어", label_visibility="collapsed")
        submitted = st.form_submit_button("가장 가까운 매뉴얼 찾기", icon=":material/search:")
    if submitted and query.strip():
        try:
            with st.spinner("질문을 임베딩해서 1,005조각과 비교하는 중..."):
                X, docs, metas, retriever = manual_vectors()
                q = np.asarray(get_embeddings().embed_query(query), dtype=float)
                scores = X @ (q / np.linalg.norm(q))
                top = np.argsort(scores)[::-1][:3]
                keyword_pages = [page_label(d) for d in retriever.keyword.search(query, k=3)]
        except Exception as e:  # 한도 초과 등
            from agent import friendly_error
            st.warning(friendly_error(e))
        else:
            rows = [{"순위": i + 1, "페이지": f"{(metas[j] or {}).get('page', 0) + 1}쪽",
                     "유사도": float(scores[j]), "내용": " ".join(docs[j].split())[:70] + "…"} for i, j in enumerate(top)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            st.caption(f"비교: 같은 질문을 키워드 검색으로 찾으면 → {', '.join(f'{p}쪽' for p in keyword_pages) or '결과 없음'}")
    st.caption("버튼을 누를 때마다 무료 임베딩 한도를 1회 사용합니다.")


def render_persona_tab():
    st.markdown(
        "AI에게 주는 **역할과 답변 규칙(시스템 프롬프트)**입니다. 아래 내용은 앱 설정 파일(`settings.py`)에서 "
        "**그대로 읽어 온 것**이라, 설정을 바꾸면 이 화면도 함께 바뀝니다."
    )
    st.code(settings.SYSTEM_PROMPT, language=None, wrap_lines=True)

    st.markdown("##### 규칙마다 넣은 이유")
    st.markdown(PERSONA_RULES)

    st.markdown("##### AI에게 실제로 전달되는 구조")
    st.mermaid_chart(PROMPT_FLOW)

    st.markdown("##### 주요 설정값")
    st.markdown(
        "| 설정 | 값 | 의미 |\n|---|---|---|\n"
        f"| 답변 모델 | `{settings.LLM_MODEL}` | 빠르고 무료 한도가 넉넉한 경량 모델 |\n"
        f"| 온도 | `{settings.LLM_TEMPERATURE}` | 0에 가까울수록 일관되고 사실 위주 (창의성↓) |\n"
        f"| 생각하기 수준 | `{settings.LLM_THINKING_LEVEL}` | 매뉴얼 정리는 깊은 추론이 필요 없어 최소화 (속도↑) |\n"
        f"| 답변 길이 상한 | `{settings.LLM_MAX_OUTPUT_TOKENS}` 토큰 | 너무 긴 답변 방지 |\n"
        f"| 검색 조각 수 | `{settings.SEARCH_K}`개 | 답변 근거로 주는 매뉴얼 조각 |\n"
        f"| 대화 기억 | 최근 `{settings.HISTORY_TURNS}`턴 | \"그럼 냉각수는?\" 같은 이어지는 질문용 |\n"
        f"| 음성 받아쓰기 힌트 | 차량 용어 `{len(GLOSSARY)}`개 | \"차체\"를 \"자체\"로 잘못 듣지 않게 |"
    )
