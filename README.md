# GENESIS AI Assistant

🔗 **배포 링크**: https://genesis-chatbot.streamlit.app/

📘 **강의 활용 가이드**: [docs/lessons/GUIDE.md](docs/lessons/GUIDE.md) — 시연 대본, 상황별 활용법, 교육 사례, 단계별 코드(태그)

제네시스 차량 매뉴얼(PDF, 767페이지)을 기반으로 질문에 답하는 **RAG 챗봇**입니다.
"AI Agent 개발을 위한 데이터 구축 전문가 과정"에서 처음으로 만들고 Streamlit Cloud에 배포한 실습 프로젝트입니다.

## 주요 기능

- **매뉴얼 검색 기반 답변 (RAG)**: 질문과 관련된 매뉴얼 내용을 벡터 검색으로 찾아 답변
- **출처 표시**: 답변 아래에 참고한 매뉴얼 페이지 번호 표시
- **스트리밍 답변**: "매뉴얼 검색 중 → 답변 작성 중" 단계 표시 후 답변이 만들어지는 대로 바로 표시
- **RAG 체인**: 매뉴얼을 먼저 검색하고 답변 모델은 1번만 호출 (Agent 대비 첫 글자 42% 빨라짐)
- **대화 기억**: 세션 단위로 이전 대화를 이어서 답변 ("그럼 냉각수는?" 같은 후속 질문)
- **음성 질문**: 입력창의 🎙 버튼으로 녹음 → Gemini가 차량 용어 힌트로 받아쓰기 → 바로 질문 (받아쓴 문장은 "음성 질문"으로 표시, 사이드바에서 확인 후 전송으로 변경 가능)
- **한도 대응**: 무료 임베딩 한도가 바닥나면 키워드 검색(BM25)으로 자동 전환, 오류는 한국어 안내
- **예시 질문**: 첫 화면에서 버튼으로 바로 질문
- **개발 과정 보기**: 제목 아래 링크로 작동 원리·**임베딩 원리**·**AI 설정(페르소나)**·개발 여정·개선 수치를 팝업으로 확인 (시연·교육용)
- **반응형 UI**: PC / 모바일 배경·레이아웃 대응, 다크 테마

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| UI / 배포 | Streamlit, Streamlit Community Cloud |
| LLM | Google Gemini `gemini-3.5-flash-lite` |
| 임베딩 | Google `gemini-embedding-001` (768차원) |
| 음성 받아쓰기 | `st.chat_input(accept_audio=True)` + Gemini (차량 용어 힌트) |
| 벡터 DB | Chroma (`chroma_db/`에 미리 생성해 저장) |
| 대체 검색 | 글자 bigram 기반 BM25 키워드 검색 (API 호출 없음) |
| 프레임워크 | LangChain (`langchain`, `langchain-google-genai`, `langchain-chroma`) |

## 동작 구조

```
[사전 준비 - 로컬에서 1회]
Genesis_2026.pdf ─▶ 분할(1,000자) ─▶ Gemini 임베딩 ─▶ chroma_db/ 저장 ─▶ git 커밋

[앱 실행]
질문(텍스트) ───────────────┐
질문(음성) ─▶ Gemini 받아쓰기 ─┴▶ 매뉴얼 검색(벡터, 한도 초과 시 키워드) ─▶ Gemini 답변 생성(1회)
                                                                  └▶ 답변 + 참고 페이지
```

Gemini 무료 티어는 임베딩 요청이 **분당 100회 / 하루 1,000회**로 제한됩니다.
그래서 앱이 시작할 때마다 PDF 전체(약 1,000개 청크)를 임베딩하지 않고,
`build_vectorstore.py`로 한 번만 만들어 둔 `chroma_db/`를 불러옵니다.

## 프로젝트 구조

```
├── streamlit_app.py       # 화면 (대화창, 음성 입력, 출처 표시)
├── settings.py            # 모델 이름, 시스템 프롬프트 등 설정
├── agent.py               # 검색 → 답변 생성(RAG 체인), 스트리밍, 오류 안내
├── retrieval.py           # 매뉴얼 검색 (벡터 + 키워드 대체 검색)
├── voice.py               # 음성 받아쓰기 (Gemini + 차량 용어 힌트)
├── showcase.py            # '개발 과정 보기' 팝업 (순서도, 개발 여정, 개선 수치)
├── showcase_theory.py     # 팝업의 임베딩 원리·AI 설정(페르소나) 탭
├── showcase_data/         # 팝업용 미리 계산한 데이터 (eval/build_embedding_map.py로 생성)
├── errors.py              # Gemini 오류 분류 (한도 초과·일시 장애 등)
├── vectorstore_config.py  # 임베딩 모델 / 벡터 DB 경로 공통 설정
├── build_vectorstore.py   # PDF → chroma_db/ 생성 스크립트 (1회 실행)
├── styles.css             # 배경 이미지 등 테마로 못 하는 스타일
├── .streamlit/config.toml # 색상·폰트 테마
├── chroma_db/             # 미리 생성한 벡터 DB
├── Genesis_2026.pdf       # 원본 매뉴얼
├── requirements.txt       # 버전 고정 (ASCII만 사용: Windows pip 인코딩 문제)
├── eval/                  # 검색·받아쓰기 평가 스크립트와 평가 세트
├── tests/                 # API 없이 도는 단위 테스트 (python -m pytest tests)
├── docs/                  # 제작 과정 기록 (기획서, 개발 일지, 결정 기록, 평가 결과)
└── practice/              # 교육 과정 실습 코드 (OpenAI 기반 초기 버전, 참고용)
    ├── 01_pdf_embedding.py   # PDF 로드 → 분할 → 임베딩 → Chroma 저장
    └── 02_rag_agent.py       # 검색 도구 + Agent 구성
```

## 로컬 실행

```powershell
# 1. 가상환경 및 패키지 설치
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. API 키 설정 (.env 파일 생성) - https://aistudio.google.com/apikey 에서 발급
GOOGLE_API_KEY=발급받은키

# 3. (chroma_db/가 없거나 PDF를 바꾼 경우에만) 벡터 DB 생성
.\.venv\Scripts\python build_vectorstore.py

# 4. 앱 실행 → http://localhost:8501
.\.venv\Scripts\streamlit run streamlit_app.py
```

- `build_vectorstore.py`는 한도에 걸려 중간에 멈추면 다시 실행할 때 이어서 진행합니다.
- PDF나 분할 설정을 바꿨다면 `chroma_db/`를 지우고 처음부터 다시 생성하세요.

## 품질 평가

```powershell
.\.venv\Scripts\python eval/run_eval.py                    # 벡터 검색 (질문 20개, Hit@3·MRR)
.\.venv\Scripts\python eval/run_eval.py --method keyword   # 키워드 검색 (API 미사용)
.\.venv\Scripts\python eval/run_voice_eval.py              # 음성 받아쓰기 (Windows TTS 음성 10개)
```

| 평가 | 결과 | 기록 |
|---|---|---|
| 벡터 검색 Hit@3 (기본) | **100% (20/20)**, MRR 0.94 | `docs/eval/2026-10-07_vector_k3.md` |
| 키워드 검색 Hit@3 (대체) | 90% (18/20) | `docs/eval/2026-10-07_keyword_k3.md` |
| 답변 속도 (첫 글자 / 완료) | 1.8초 / 3.9초 (개선 전 3.1초 / 6.1초) | `docs/eval/2026-10-07_latency_*.md` |
| 음성 받아쓰기 (용어 힌트) | 10/10 (힌트 없음 6/10) | `docs/eval/2026-10-07_voice.md` |

## 무료 한도 (Gemini Free Tier)

| 항목 | 한도 | 이 앱에서 |
|---|---|---|
| 답변 생성 `gemini-3.5-flash-lite` | 분당 15회 | 질문 1개당 1회 (음성이면 +1회) |
| 임베딩 `gemini-embedding-001` | 분당 100회, 하루 1,000회 | 질문 1개당 1~3회 (바닥나면 키워드 검색) |

## 배포 (Streamlit Community Cloud)

1. GitHub 저장소를 연결하고 메인 파일을 `streamlit_app.py`로 지정
2. **Settings → Secrets**에 API 키 등록
   ```toml
   GOOGLE_API_KEY = "발급받은키"
   ```
3. `chroma_db/`가 커밋되어 있어야 합니다 (앱은 임베딩을 새로 만들지 않음)

## 개발 이력

| 단계 | 내용 |
|---|---|
| 01 | 1차 바이브 코딩: PDF 임베딩 + Chroma DB |
| 02 | Streamlit Cloud 배포용으로 전환 |
| 03 | UI/UX 개선, 대화 기억 |
| 04~06 | 음성 인식 기능 및 모바일 호환성 개선 |
| 07 | 음성 비서 온보딩 토글, 모바일 UI 조정 |
| 08 | OpenAI → Google Gemini 전환, 벡터 DB 사전 생성 방식으로 변경 |
| 09 | 음성 인식 오버레이/취소 버그 수정, 프로젝트 파일 정리 |
| 10 | 데모 고도화: 음성 받아쓰기 개편, 출처·스트리밍, 대체 검색, 코드 분리, 평가 세트, 제작 기록 |
| 11 | 답변 속도 개선: Agent → RAG 체인 (첫 글자 3.1초 → 1.8초) |
| 12 | 개발 과정 보기 팝업 (시연·교육용 시각화) |
| 13 | 임베딩 원리·AI 설정(페르소나) 탭 |

자세한 과정은 [`CHANGELOG.md`](CHANGELOG.md)와 [`docs/`](docs/)를 참고하세요.
