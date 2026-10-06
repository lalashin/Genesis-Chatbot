"""
Gemini API 오류 분류 (한 곳에서만 판별해 대체 검색·사용자 안내·빌드 스크립트가 같은 기준을 쓰게 합니다).

오류 메시지 안의 숫자(토큰 수, 요청 ID 등)에 '429'가 섞여도 오분류되지 않도록
상태 코드는 단어 경계로, 상태 이름은 정확한 문자열로 찾습니다.
"""
import re


def classify(e: Exception) -> str | None:
    """'quota_day' | 'quota_minute' | 'unavailable' | 'not_found' | 'auth' | None"""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or re.search(r"\b429\b", msg):
        return "quota_day" if "PerDay" in msg else "quota_minute"
    if "UNAVAILABLE" in msg or re.search(r"\b503\b", msg):
        return "unavailable"
    if "NOT_FOUND" in msg or re.search(r"\b404\b", msg):
        return "not_found"
    if "PERMISSION_DENIED" in msg or "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "auth"
    return None


def is_quota_or_unavailable(e: Exception) -> bool:
    return classify(e) in ("quota_day", "quota_minute", "unavailable")


def quota_ids(e: Exception) -> list[str]:
    return re.findall(r"'quotaId': '([^']+)'", str(e))
