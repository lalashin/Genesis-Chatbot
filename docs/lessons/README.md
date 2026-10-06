# 교육 자료 목차 (초안)

> "AI 채팅만으로 만든 챗봇을 제대로 된 데모로 고도화하기"
> 각 차시는 [템플릿](../templates/lesson.md) 형식으로 채웁니다. 아래는 차시 구성과 연결 자료입니다.
> 차시·시수 설계는 `pbl-curriculum-design` 스킬로 구체화할 수 있습니다.

## 학습 흐름

```
[1부] 바이브 코딩으로 첫 버전 만들기 (01~07)  →  [2부] 점검·전환 (08~09)  →  [3부] 데모 고도화 (10)
       "동작하는 것"                                  "왜 깨졌나"                     "근거로 증명하기"
```

## 차시 구성

| 차시 | 주제 | 핵심 개념 | 연결 코드 | 실제 문제 사례 (교육 소재) |
|---|---|---|---|---|
| 1 | RAG 기본: PDF → 청크 → 임베딩 → 벡터 DB | 청킹, 임베딩, 유사도 검색 | `practice/01_pdf_embedding.py` | 분할 구분자 `"\\n"` 버그 (2025-12 회고) |
| 2 | Agent와 도구 | `create_agent`, tool calling | `practice/02_rag_agent.py` | LangChain 버전 차이 (디버그 파일 회고) |
| 3 | Streamlit 챗봇과 배포 | 세션 상태, Secrets, Cloud 배포 | 브랜치 `02-...` | requirements UTF-16, venv 커밋, 비공개 저장소 배포 실패 |
| 4 | 음성 인터페이스 1: 브라우저 음성 인식 | Web Speech API, iframe | 브랜치 `04~07` | 오버레이 안 닫힘(iframe 재생성), 중복 전송 → [devlog](../devlog/2026-10-06_gemini-migration.md) |
| 5 | LLM 공급자 전환과 무료 한도 설계 | 모델 교체, 레이트 리밋, 사전 임베딩 | 커밋 `d34b7ce4` | 모델 404, 임베딩 429 → [결정 001](../decisions/001-gemini.md), [002](../decisions/002-prebuilt-vectorstore.md) |
| 6 | 검색 품질 평가 | 평가 세트, Hit@k, MRR | `eval/run_eval.py` | 정답 범위를 좁게 잡아 70% → 90% (측정 기준의 중요성) |
| 7 | 음성 인터페이스 2: 맥락 있는 받아쓰기 | 멀티모달 LLM STT, 도메인 용어 힌트 | `voice.py` | "차체→자체" 재현과 해결 6/10 → 10/10 → [결정 004](../decisions/004-voice-audio-input.md) |
| 8 | 데모 안정성과 신뢰성 | 출처 표시, 스트리밍, 대체 검색, 오류 안내 | `agent.py`, `retrieval.py` | 한도 바닥 시 키워드 검색 전환, `<br>` 노출 |
| 9 | 코드 구조와 배포 체크리스트 | 모듈 분리, 테마, 버전 고정 | 브랜치 `10-demo-upgrade` | requirements 한글 주석 → Windows pip cp949 오류 |

## 차시마다 넣을 실습 아이디어
- **비교 실습**: 같은 질문을 1부 버전과 3부 버전에 던져 차이(출처, 정확도, 오류 처리) 관찰
- **평가 실습**: `chunk_size`를 바꿔 벡터 DB를 다시 만들고 Hit@3 변화 측정
- **프롬프트 실습**: `settings.SYSTEM_PROMPT`의 "수치를 그대로 제시" 규칙을 지웠을 때 답변 비교
- **용어 힌트 실습**: `voice.GLOSSARY`를 비우고 `eval/run_voice_eval.py` 재측정

## 배포 전 체크리스트 (실제 사고에서 나온 것)
- [ ] `.env`, `secrets.toml`이 `.gitignore`에 있는가 (API 키 커밋 사고)
- [ ] `venv/` 같은 가상환경 폴더가 커밋되지 않았는가
- [ ] `requirements.txt`가 UTF-8(가능하면 ASCII)인가 (UTF-16, cp949 사고)
- [ ] **빈 가상환경에 requirements만 설치해서 앱이 뜨는가**
- [ ] 모델 이름을 실제 호출로 확인했는가 (404 사고)
- [ ] 무료 한도를 넘었을 때 앱이 어떻게 동작하는가 (429 사고)
- [ ] 저장소 공개 범위와 배포 서비스 권한이 맞는가 (clone 실패 사고)
