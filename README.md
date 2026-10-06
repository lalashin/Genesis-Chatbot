# GENESIS AI Assistant

🔗 **배포 링크**: https://genesis-chatbot.streamlit.app/

제네시스 차량 매뉴얼(PDF, 767페이지)을 기반으로 질문에 답하는 **RAG 챗봇**입니다.
"AI Agent 개발을 위한 데이터 구축 전문가 과정"에서 처음으로 만들고 Streamlit Cloud에 배포한 실습 프로젝트입니다.

## 주요 기능

- **매뉴얼 검색 기반 답변 (RAG)**: 질문과 관련된 매뉴얼 내용을 벡터 검색으로 찾아 답변
- **Agent 구조**: LangChain `create_agent` + 매뉴얼 검색 도구(`search_manual`)
- **대화 기억**: 세션 단위로 이전 대화를 이어서 답변
- **음성 질문**: 🎤 버튼으로 말하면 자동으로 질문 전송 (Web Speech API, 한국어)
- **온보딩 토글**: 음성 비서 활성화 시 마이크 권한을 미리 요청
- **반응형 UI**: PC / 모바일 배경·레이아웃 대응, 다크 테마

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| UI / 배포 | Streamlit, Streamlit Community Cloud |
| LLM | Google Gemini `gemini-3.5-flash-lite` |
| 임베딩 | Google `gemini-embedding-001` (768차원) |
| 벡터 DB | Chroma (`chroma_db/`에 미리 생성해 저장) |
| 프레임워크 | LangChain (`langchain`, `langchain-google-genai`, `langchain-chroma`) |

## 동작 구조

```
[사전 준비 - 로컬에서 1회]
Genesis_2026.pdf ─▶ 분할(1,000자) ─▶ Gemini 임베딩 ─▶ chroma_db/ 저장 ─▶ git 커밋

[앱 실행]
질문(텍스트/음성) ─▶ Agent(Gemini) ─▶ search_manual 도구 ─▶ chroma_db 검색 ─▶ 답변
```

Gemini 무료 티어는 임베딩 요청이 **분당 100회 / 하루 1,000회**로 제한됩니다.
그래서 앱이 시작할 때마다 PDF 전체(약 1,000개 청크)를 임베딩하지 않고,
`build_vectorstore.py`로 한 번만 만들어 둔 `chroma_db/`를 불러옵니다.

## 프로젝트 구조

```
├── streamlit_app.py       # 챗봇 앱 (UI, Agent, 음성 인식)
├── vectorstore_config.py  # 임베딩 모델 / 벡터 DB 경로 공통 설정
├── build_vectorstore.py   # PDF → chroma_db/ 생성 스크립트 (1회 실행)
├── chroma_db/             # 미리 생성한 벡터 DB
├── Genesis_2026.pdf       # 원본 매뉴얼
├── requirements.txt
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
