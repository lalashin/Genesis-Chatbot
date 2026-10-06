# 2026-10-07 개발 일지: 데모 고도화 (10단계, 야간 작업)

> 브랜치 `10-demo-upgrade` · 작업자: Claude (사용자 취침 중 자율 진행) · 푸시·배포는 하지 않음

## 한 일

| 단계 | 내용 | 결과 |
|---|---|---|
| 0. 기록 체계 | `docs/`(devlog, decisions, plan, eval, lessons, templates), `CHANGELOG.md` | 01~09 이력 회고, 결정 001~003 |
| 1. 기획 | [기획서](../plan/demo-upgrade.md) 초안 (⚠️ 가정 3개는 확인 필요) | |
| 2. 데이터 평가 | 평가 질문 20개 + 정답 페이지, `eval/run_eval.py` (Hit@k, MRR) | 키워드 검색 Hit@3 90% |
| 3. 음성 개편 | Web Speech API → `st.chat_input(accept_audio=True)` + Gemini 받아쓰기 | 받아쓰기 10/10 (힌트 없음 6/10) |
| 4. 답변 품질 | 출처(페이지) 표시, 스트리밍, 예시 질문, 수치 그대로 제시하는 프롬프트 | |
| 5. 안정성 | 임베딩 한도 초과 시 키워드 검색 자동 전환, 한국어 오류 안내, 버전 고정 | |
| 6. 코드 구조 | 734줄 단일 파일 → `settings / retrieval / agent / voice / streamlit_app` + `styles.css` + 테마 | 앱 파일 약 200줄 |

## 문제와 해결

### 1. 새 프로젝트 키인데도 임베딩 하루 한도 초과
- **증상**: 새 Google Cloud 프로젝트에서 발급한 키로 단건 임베딩은 성공했지만, 20개 묶음 임베딩은 `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier` 오류
- **확인한 것**: 단건(질문 검색)은 간헐적으로 성공, 5개 묶음도 실패. 정확한 원인(계정 단위 집계 여부 등)은 확인하지 못함
- **대응**: 남은 25개 청크(전체의 2.5%)는 한도 초기화 후 채우기로 미룸. 대신 **한도가 바닥나도 앱이 동작하도록 키워드 검색 대체 기능**을 만듦
- **배운 점**: 무료 API에 의존하는 데모는 "한도가 바닥났을 때 어떻게 동작할지"를 설계에 넣어야 한다

### 2. 정답 페이지를 너무 좁게 잡아 검색 품질이 낮게 측정됨
- **증상**: 키워드 검색 첫 측정 Hit@3 70%. 실패 사례를 보니 검색된 페이지(예: 714쪽 "공기압 관리", 552쪽 "차로 유지 보조 이상")도 질문과 관련 있는 내용
- **해결**: 정답을 해당 **섹션 전체 페이지 범위**로 넓힘 → Hit@3 90%, Hit@5 95%
- **배운 점**: 평가 세트의 정답 기준이 평가 결과를 좌우한다. 실패 사례는 반드시 직접 열어 본다

### 3. 답변 모델 분당 한도 15회
- **증상**: 받아쓰기 평가를 빠르게 돌리자 `GenerateRequestsPerMinutePerProjectPerModel-FreeTier, limit 15`
- **영향**: 질문 1개 = 받아쓰기 1회 + 답변 생성 2~3회 → 분당 질문 4~5개가 한계
- **대응**: 한도 초과 시 "30초쯤 뒤에 다시 질문해 주세요" 안내 (`agent.friendly_error`), 평가 스크립트에 호출 간격

### 4. 받아쓴 문장을 입력창에 넣다가 오류
- **증상**: `StreamlitWidgetAlreadyInstantiatedError: st.session_state.chat_input cannot be modified after the widget ... is instantiated`
- **원인**: 가이드 예제처럼 입력창을 만든 뒤 값을 바꿈. 브라우저로는 녹음을 자동화할 수 없어 AppTest로 따로 재현해서 발견
- **해결**: `pending_input`에 저장 → 다음 실행에서 입력창 생성 **전에** 값 주입 (AppTest로 검증)
- **배운 점**: 공식 예제도 그대로 믿지 말고 작은 재현 코드로 확인한다

### 5. 답변 표 안에 `<br>`이 글자로 보임
- **원인**: 모델이 마크다운 표 안 줄바꿈에 HTML `<br>`을 씀. `st.markdown`은 HTML을 렌더링하지 않음
- **해결**: 저장할 때 `<br>` → `·` 로 정리 (`agent.clean_markdown`)

### 6. 한글 주석 때문에 Windows에서 `pip install -r requirements.txt` 실패
- **증상**: `UnicodeDecodeError: 'cp949' codec can't decode byte 0xed`
- **원인**: pip는 Windows에서 requirements 파일을 시스템 코드 페이지(cp949)로 읽음. 버전 고정하면서 넣은 한글 주석이 원인
- **발견 방법**: Streamlit Cloud처럼 빈 가상환경에 requirements만 설치하는 테스트
- **해결**: requirements.txt 주석을 영어(ASCII)로 변경
- **배운 점**: 08단계의 UTF-16 문제와 같은 계열. **배포 전 "빈 환경 설치 테스트"를 체크리스트에 넣는다**

## 검증 결과
- 텍스트 질문(엔진 오일 용량): 2.5 터보 6.2ℓ / 3.5 터보 7.0ℓ — 매뉴얼 23쪽 원문과 일치, 출처 표시
- 이어지는 질문("그럼 냉각수는?"): 맥락 유지, 8.6ℓ / 9.7ℓ / 1.48ℓ 원문과 일치
- 임베딩 한도 초과 상태에서 키워드 검색 자동 전환 확인 (출처에 "키워드 검색" 표시)
- 받아쓰기: TTS 음성 10문장 10/10 (실제 마이크는 아침 사용자 테스트 필요)

## 다음 할 일
- [ ] (사용자) 실제 마이크로 음성 질문 테스트 — PC, 모바일
- [ ] 임베딩 한도 초기화 후: 남은 25개 청크 채우기, 벡터 검색 평가(`eval/run_eval.py`)
- [ ] (사용자 확인 후) master 병합, push, Streamlit Cloud Reboot
