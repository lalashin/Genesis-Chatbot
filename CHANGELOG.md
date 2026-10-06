# 변경 이력

단계별 주요 변경 사항입니다. 자세한 과정은 [`docs/devlog/`](docs/devlog/)를 참고하세요.

## [10] 데모 고도화 — 진행 중 (브랜치 `10-demo-upgrade`)
- 음성 질문: 브라우저 음성 인식 → 입력창 녹음 + Gemini 받아쓰기(차량 용어 힌트), 받아쓴 문장 확인 후 전송
- 답변 아래 참고한 매뉴얼 페이지 표시, 답변 스트리밍, 첫 화면 예시 질문
- 매뉴얼의 수치·절차를 그대로 제시하도록 프롬프트 개선
- 임베딩 무료 한도가 바닥나면 키워드 검색으로 자동 전환, 오류를 한국어 안내로 표시
- 코드 분리: settings / retrieval / agent / voice / streamlit_app, 테마는 `.streamlit/config.toml`
- 패키지 버전 고정 (`requirements.txt`는 ASCII만: Windows pip 인코딩 문제)
- 평가 세트: 검색 20문항(`eval/run_eval.py`), 음성 받아쓰기 10문항(`eval/run_voice_eval.py`)
- 제작 과정 기록 체계 추가 (`docs/`: 기획서, 개발 일지, 결정 기록, 평가, 교육 자료)

## [09] 2026-10-07 — 정리와 음성 버그 수정
- 음성 질문 후 오버레이가 닫히지 않고 취소가 안 되던 문제 수정
- 음성 질문이 한 문장인데 여러 번 전송되던 문제 수정
- 불필요한 파일 정리, FastAPI 버전(`server.py`, `index.html`) 제거, 실습 코드 `practice/`로 이동
- README를 프로젝트 소개 문서로 새로 작성

## [08] 2026-10-06 — Google Gemini 전환
- LLM `gpt-4o-mini` → `gemini-3.5-flash-lite`, 임베딩 → `gemini-embedding-001`(768차원)
- 벡터 DB를 미리 생성해 커밋 (`build_vectorstore.py`, `chroma_db/`) → 앱 시작 시 임베딩 없음
- 문서 분할 줄바꿈 버그, 빈 답변으로 대화가 깨지는 문제 수정
- `requirements.txt` 인코딩(UTF-16 → UTF-8) 수정, `venv/` 추적 해제

## [01~07] 2025-12 — 첫 제작과 배포
- PDF 매뉴얼 RAG + Agent 챗봇, Streamlit Cloud 배포
- 다크 테마 UI, 대화 기억, 음성 질문, 모바일 대응, 음성 비서 온보딩 토글
