# 변경 이력

단계별 주요 변경 사항입니다. 자세한 과정은 [`docs/devlog/`](docs/devlog/)를 참고하세요.

## [10] 데모 고도화 — 진행 중 (브랜치 `10-demo-upgrade`)
- 제작 과정 기록 체계 추가 (`docs/`: 기획서, 개발 일지, 결정 기록, 평가, 교육 자료 템플릿)

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
