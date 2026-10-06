# 2026-10-06 ~ 10-07 개발 일지: Gemini 전환과 음성 버그 수정 (08~09단계)

## 한 일
- OpenAI → Google Gemini 전환 (LLM, 임베딩, API 키) — `d34b7ce4`
- 벡터 DB를 앱 시작 시 생성 → 미리 생성(`build_vectorstore.py`) 후 로드로 변경
- 프로젝트 파일 정리, README 새로 작성 — `d05e5443`
- 음성 인식 오버레이/취소/중복 전송 버그 수정 — `40f20ddb`
- GitHub 원격 README 변경과 병합 후 push — `71214a3f`

## 문제와 해결

### 1. 모델 404: `gemini-2.5-flash is no longer available to new users`
- **증상**: 질문하면 `404 NOT_FOUND` 오류
- **원인**: 새로 발급한 키(신규 사용자)에서는 해당 모델 사용 불가. 전환 시 모델 이름을 문서로 확인하지 않고 기억에 의존했다.
- **해결**: 후보 모델을 실제로 호출해 비교 → `gemini-3.5-flash-lite` 선택 (`gemini-3.8-flash`는 503 과부하)
- **배운 점**: 모델 이름은 짐작하지 말고 **실제 호출로 확인**한다.

### 2. 임베딩 429: 무료 티어 한도
- **증상**: 앱 시작 시 `RESOURCE_EXHAUSTED` → 앱이 뜨지 않음
- **원인**: 무료 티어 임베딩 한도 **분당 100회 / 하루 1,000회**. PDF 767페이지 = 1,005개 청크를 앱 시작마다 임베딩.
- **해결**: 로컬에서 한 번만 천천히 임베딩해 `chroma_db/`에 저장하고 커밋, 앱은 로드만 → [결정 002](../decisions/002-prebuilt-vectorstore.md)
- **추가 발견**: 한도는 **키가 아니라 Google Cloud 프로젝트 단위**. 같은 프로젝트에서 키만 바꾸면 한도도 그대로.
- **배운 점**: 무료 API로 데모를 만들 때는 "앱 시작 시 대량 호출" 구조를 피한다.

### 3. 음성 오버레이가 닫히지 않고 취소 버튼이 동작하지 않음
- **증상**: 질문을 말한 뒤 "말씀하세요..." 화면이 남음, 취소를 눌러도 반응 없음
- **원인 찾기**: 브라우저에서 직접 재현
  1. 질문 전송 시 Streamlit rerun → `components.html` iframe이 **새로 만들어짐**을 확인 (`sameIframe: false`)
  2. iframe 안에서 등록한 이벤트 핸들러가 iframe 제거 후 실행되는지 실험 → **실행 안 됨** (`firedAfterIframeRemoved: 0`)
- **해결**: 음성 로직을 부모 페이지에 `<script>`로 한 번만 설치해 rerun 후에도 유지
- **배운 점**: 증상만 보고 고치지 말고 **가설을 세워 재현으로 증명**한 뒤 고친다.

### 4. 음성 질문이 "타이어", "타이어가", "타이어가 귀여운"... 여러 개로 전송
- **증상**: 한 문장을 말했는데 질문이 10개 가까이 전송됨
- **원인**: 모바일 Chrome 등은 `interimResults=false`여도 `onresult`를 여러 번 보냄. 이전에는 3번 버그(핸들러가 죽음) 때문에 첫 결과만 처리되어 **우연히 가려져 있었다**.
- **해결**: 듣기 1회당 마지막 문장만 1번 전송 (onend 또는 1.2초 무응답 시)
- **배운 점**: 버그 하나를 고치면 그 버그에 가려져 있던 다른 버그가 드러날 수 있다.

### 5. Streamlit Cloud 배포 실패: `Failed to download the sources`
- **증상**: 앱 접속 시 "Oh no. Error running app"
- **원인**: GitHub 저장소가 비공개라 Streamlit Cloud가 clone하지 못함 (로그인 없이 접속 시 404로 확인)
- **해결**: 저장소 공개 전환 (Reboot 필요)

## 결정한 것
- [001. LLM을 OpenAI에서 Google Gemini로 전환](../decisions/001-gemini.md)
- [002. 벡터 DB를 미리 생성해 커밋](../decisions/002-prebuilt-vectorstore.md)
- [003. 음성 로직을 부모 페이지에 설치](../decisions/003-voice-parent-script.md)

## 다음 할 일
- 고도화 0~8단계 → [기획서](../plan/demo-upgrade.md)
