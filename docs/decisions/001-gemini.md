# 001. LLM과 임베딩을 OpenAI에서 Google Gemini로 전환

- **날짜**: 2026-10-06
- **상태**: 채택

## 배경
OpenAI API 키가 만료/폐기되어 챗봇이 동작하지 않았다. 실습용 데모로 계속 운영하려면 비용 부담이 적은 대안이 필요했다.

## 선택지
| 선택지 | 장점 | 단점 |
|---|---|---|
| OpenAI 유료 키 재발급 | 코드 변경 없음 | 사용량만큼 과금 |
| **Google Gemini 무료 티어** | 무료, LangChain 지원(`langchain-google-genai`) | 요청 한도(임베딩 분당 100/하루 1,000), 모델 교체가 잦음 |

## 결정
- LLM: `gemini-3.5-flash-lite` (처음 선택한 `gemini-2.5-flash`는 신규 키에서 404)
- 임베딩: `gemini-embedding-001`, `output_dimensionality=768`

## 이유
- 실습 데모는 사용량이 적어 무료 한도로 충분하다.
- flash-lite는 응답이 빠르고 한도가 넉넉해 시연에 적합하다. 답변 품질도 사용자 테스트에서 만족.
- 768차원으로 줄여 벡터 DB 용량을 1/4로 줄였다(커밋해도 부담 없음).

## 영향
- 임베딩 모델이 바뀌어 기존 OpenAI용 벡터 DB를 재사용할 수 없다 → [002](002-prebuilt-vectorstore.md)
- 한도 초과(429)에 대한 처리가 필요하다.
