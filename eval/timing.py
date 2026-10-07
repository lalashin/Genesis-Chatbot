"""
답변 지연 측정: 질문마다 검색 시간, 첫 글자까지, 답변 완료까지 걸린 시간을 잽니다.

실행: python eval/timing.py [--save docs/eval/...md] [질문 ...]
(질문을 주지 않으면 기본 4개. 분당 한도 15회 때문에 질문 사이에 쉬어 갑니다)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

DEFAULT_QUESTIONS = [
    "타이어 공기압은 얼마나 넣어야 해?",
    "엔진이 과열되면 어떻게 해?",
    "스마트 크루즈 컨트롤 사용법 알려줘",
    "주행 중에 차가 고장 나면 어떻게 해야 해?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--save")
    args = parser.parse_args()
    load_dotenv()

    from agent import build_assistant
    from retrieval import ManualRetriever

    assistant = build_assistant(ManualRetriever())
    rows = []
    for i, q in enumerate(args.questions or DEFAULT_QUESTIONS):
        if i:
            time.sleep(10)
        messages = [{"role": "user", "content": q}]
        t0 = time.perf_counter()
        result = assistant.search(messages)
        t_search = time.perf_counter() - t0
        first, chars = None, 0
        for text in assistant.stream(messages, result):
            if first is None and text.strip():
                first = time.perf_counter() - t0
            chars += len(text)
        total = time.perf_counter() - t0
        rows.append((q, t_search, first or total, total, chars, result.method))
        print(f"{q}\n   검색 {t_search:.2f}s ({result.method}) · 첫 글자 {first or total:.2f}s · 완료 {total:.2f}s · {chars}자")

    n = len(rows)
    avg = lambda idx: sum(r[idx] for r in rows) / n  # noqa: E731
    summary = f"평균: 검색 {avg(1):.2f}s · 첫 글자 {avg(2):.2f}s · 완료 {avg(3):.2f}s · {avg(4):.0f}자"
    print("\n" + summary)
    if args.save:
        lines = [f"# 답변 지연 측정 ({time.strftime('%Y-%m-%d %H:%M')})", "", f"**{summary}**", "",
                 "| 질문 | 검색 | 첫 글자 | 완료 | 답변 길이 |", "|---|---|---|---|---|"]
        lines += [f"| {q} | {s:.2f}s | {f:.2f}s | {t:.2f}s | {c}자 |" for q, s, f, t, c, _ in rows]
        open(args.save, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
