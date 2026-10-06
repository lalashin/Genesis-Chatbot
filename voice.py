"""
음성 받아쓰기 (Speech-to-Text).

브라우저 내장 음성 인식(Web Speech API)은 앱의 주제를 몰라
"차체"를 "자체"처럼 일상어로 알아듣는 문제가 있었습니다.
Gemini에 녹음 파일과 함께 '제네시스 차량 매뉴얼 질문'이라는 맥락과 용어 목록을 주면
비슷한 발음의 단어를 차량 용어 쪽으로 바르게 받아씁니다. → docs/decisions/004-voice-audio-input.md
"""
from functools import lru_cache

from google import genai
from google.genai import types

from settings import STT_MODEL

# 매뉴얼에 자주 나오는 용어 (받아쓰기 힌트)
GLOSSARY = [
    "제네시스", "차체", "타이어", "공기압", "TPMS", "펑크", "휠", "와이퍼", "블레이드", "워셔액",
    "엔진 오일", "냉각수", "브레이크액", "배터리", "점프 스타트", "퓨즈", "퓨즈박스", "전구",
    "하이패스", "스마트 키", "디지털 키", "원격 시동", "시동 버튼", "오토 홀드", "전자식 파킹 브레이크",
    "스마트 크루즈 컨트롤", "차로 유지 보조", "고속도로 주행 보조", "전방 충돌방지 보조",
    "후측방 충돌방지 보조", "주차 충돌방지 보조", "서라운드 뷰 모니터", "헤드업 디스플레이", "클러스터",
    "인포테인먼트", "열선 시트", "통풍 시트", "스티어링 휠", "공조", "에어컨", "연료 주입구", "주유구",
    "경고등", "트렁크", "테일게이트", "선루프", "4륜구동", "주행 모드", "타이어 체인", "견인",
]

STT_PROMPT = (
    "다음 녹음은 제네시스 차량 사용자가 차량 매뉴얼 챗봇에게 묻는 한국어 질문입니다.\n"
    "들린 말을 한국어 문장 한 줄로 정확히 받아쓰세요.\n"
    "- 발음이 비슷하면 아래 차량 용어 쪽으로 해석하세요 (예: '자체'보다 '차체').\n"
    "- 답변하지 말고 받아쓴 문장만 출력하세요. 따옴표나 설명을 붙이지 마세요.\n"
    "- 말소리가 없거나 알아들을 수 없으면 아무것도 출력하지 마세요.\n"
    f"차량 용어: {', '.join(GLOSSARY)}"
)


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client()  # 환경변수 GOOGLE_API_KEY 사용


def transcribe(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """녹음 파일을 받아 한국어 문장 한 줄로 돌려줍니다. 인식 실패 시 빈 문자열."""
    response = _client().models.generate_content(
        model=STT_MODEL,
        contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), STT_PROMPT],
        config=types.GenerateContentConfig(temperature=0),
    )
    text = (response.text or "").strip().strip('"“”\'')
    return text.splitlines()[0].strip() if text else ""
