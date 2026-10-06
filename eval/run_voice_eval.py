"""
음성 받아쓰기 평가: Windows TTS(한국어 Heami)로 질문 음성을 만들고 받아쓰기 정확도를 측정합니다.

실행: python eval/run_voice_eval.py [--save docs/eval/voice.md]

- 용어 힌트 프롬프트 사용(voice.transcribe) vs 힌트 없이 받아쓰기를 비교합니다.
- 주의: TTS 음성은 사람 목소리보다 또렷하므로 실제 정확도보다 높게 나올 수 있습니다.
  실제 마이크 테스트로 꼭 확인하세요.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from google.genai import types  # noqa: E402

import voice  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAIN_PROMPT = "이 한국어 녹음을 그대로 받아쓰세요. 받아쓴 문장만 출력하세요."


def synthesize(text: str, path: str):
    # 문장·경로는 명령 문자열에 끼워 넣지 않고 환경변수로 넘깁니다 (따옴표가 있어도 안전)
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SelectVoice('Microsoft Heami Desktop');"
        "$s.SetOutputToWaveFile($env:TTS_PATH);"
        "$s.Speak($env:TTS_TEXT); $s.Dispose()"
    )
    env = {**os.environ, "TTS_TEXT": text, "TTS_PATH": path}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env)


def plain_transcribe(audio: bytes) -> str:
    r = voice._client().models.generate_content(
        model=voice.STT_MODEL,
        contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"), PLAIN_PROMPT],
        config=types.GenerateContentConfig(temperature=0),
    )
    return (r.text or "").strip()


def norm(s: str) -> str:
    return re.sub(r"[\s.,?!~\"'“”]", "", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()
    load_dotenv()

    items = json.load(open(os.path.join(HERE, "voice_questions.json"), encoding="utf-8"))
    rows, ok_hint, ok_plain = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for it in items:
            wav = os.path.join(tmp, f"{it['id']}.wav")
            synthesize(it["text"], wav)
            audio = open(wav, "rb").read()
            hint = voice.transcribe(audio)
            time.sleep(4)
            plain = plain_transcribe(audio)
            h_ok, p_ok = norm(hint) == norm(it["text"]), norm(plain) == norm(it["text"])
            ok_hint += h_ok
            ok_plain += p_ok
            rows.append((it["text"], hint, h_ok, plain, p_ok))
            print(f"{'O' if h_ok else 'X'}/{'O' if p_ok else 'X'} {it['text']} | 힌트: {hint} | 힌트없음: {plain}")
            time.sleep(9)  # 분당 15회 한도: 항목당 2회 호출

    n = len(items)
    summary = f"용어 힌트 사용: {ok_hint}/{n} 정확, 힌트 없음: {ok_plain}/{n} 정확 (띄어쓰기·문장부호 무시)"
    print("\n" + summary)
    if args.save:
        lines = [f"# 음성 받아쓰기 평가 ({time.strftime('%Y-%m-%d %H:%M')})", "",
                 f"**{summary}**", "", "> TTS(Microsoft Heami) 합성 음성 기준. 실제 사람 목소리는 더 어려울 수 있습니다.", "",
                 "| 원문 | 용어 힌트 사용 | | 힌트 없음 | |", "|---|---|---|---|---|"]
        for text, hint, h_ok, plain, p_ok in rows:
            lines.append(f"| {text} | {hint} | {'O' if h_ok else 'X'} | {plain} | {'O' if p_ok else 'X'} |")
        open(args.save, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
