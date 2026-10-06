"""
검색 품질 평가: 질문별로 정답 페이지가 검색 결과 상위 k개 안에 있는지 측정합니다.

실행 예:
  python eval/run_eval.py                  # 벡터 검색, k=3 (임베딩 API 사용: 질문당 1회)
  python eval/run_eval.py --method keyword  # 키워드 검색 (API 사용 안 함)
  python eval/run_eval.py --k 5 --save docs/eval/result.md

지표
  Hit@k : 정답 페이지가 상위 k개 결과 중 하나라도 있으면 성공으로 보는 비율
  MRR   : 첫 정답이 몇 번째에 나왔는지의 역수 평균 (1등이면 1, 2등이면 0.5 ...)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from retrieval import ManualRetriever, page_label  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vector", "keyword"], default="vector")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--save", help="결과를 마크다운으로 저장할 경로")
    args = parser.parse_args()

    load_dotenv()
    questions = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
    retriever = ManualRetriever()

    rows, hits, rr_sum = [], 0, 0.0
    for q in questions:
        result = retriever.search(q["question"], k=args.k, force_keyword=args.method == "keyword")
        if args.method == "vector" and result.method != "vector":
            raise SystemExit("임베딩 API를 쓸 수 없어(한도 초과 등) 벡터 검색 평가를 중단합니다.")
        found = [page_label(d) for d in result.docs]
        rank = next((i + 1 for i, p in enumerate(found) if p in q["pages"]), None)
        hits += rank is not None
        rr_sum += 1 / rank if rank else 0
        rows.append((q, found, rank))
        print(f"{'O' if rank else 'X'} {q['id']:>2}. {q['question']}  정답 {q['pages']}  검색 {found}")
        if args.method == "vector":
            time.sleep(0.7)  # 분당 한도(100회) 여유

    n = len(questions)
    summary = f"{args.method} 검색, k={args.k}: Hit@{args.k} = {hits}/{n} ({hits / n:.0%}), MRR = {rr_sum / n:.2f}"
    print("\n" + summary)

    if args.save:
        lines = [f"# 검색 평가 결과 ({time.strftime('%Y-%m-%d %H:%M')})", "", f"**{summary}**", "",
                 "| # | 질문 | 정답 페이지 | 검색된 페이지 | 결과 |", "|---|---|---|---|---|"]
        for q, found, rank in rows:
            lines.append(f"| {q['id']} | {q['question']} | {q['pages']} | {found} | {f'{rank}위' if rank else '실패'} |")
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        open(args.save, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
