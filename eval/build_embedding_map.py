"""
'개발 과정 보기 > 임베딩 원리'의 뜻 지도 데이터를 만듭니다.

매뉴얼 6개 주제의 조각(벡터 DB에 저장된 임베딩)과, 매뉴얼 단어를 일부러 피해서 쓴 질문 6개를
임베딩한 뒤, 질문×조각 코사인 유사도(768차원 그대로)를 showcase_data/embedding_map.json 에 저장합니다.

처음에는 768차원을 2차원으로 압축(PCA)한 산점도를 그렸지만, Gemini는 질문용·문서용 임베딩을
다르게 만들어서 압축하면 "주제"보다 "질문이냐 문서냐"로 나뉘어 보였습니다(오해를 부르는 그림).
그래서 압축하지 않은 실제 유사도를 히트맵으로 보여 줍니다.
앱은 이 파일만 읽으므로 팝업을 열 때 API를 쓰지 않습니다.

실행: python eval/build_embedding_map.py   (질문 임베딩 6회 사용)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

from vectorstore_config import get_embeddings  # noqa: E402

# (주제, 매뉴얼 PDF 페이지(1부터), 매뉴얼 단어를 피한 질문)
TOPICS = [
    ("타이어 공기압", 18, "바퀴에 바람은 얼마나 넣어야 돼?"),
    ("엔진 과열", 646, "엔진이 너무 뜨거워졌어"),
    ("하이패스", 219, "톨게이트 요금이 자동으로 나가는 카드"),
    ("스마트 크루즈", 519, "앞차를 따라 알아서 속도를 맞춰 가는 기능"),
    ("와이퍼", 705, "앞유리 닦는 고무 갈기"),
    ("디지털 키", 167, "스마트폰으로 차 문 열기"),
]
OUT = os.path.join(ROOT, "showcase_data", "embedding_map.json")


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def main():
    load_dotenv()
    import chromadb

    col = chromadb.PersistentClient(os.path.join(ROOT, "chroma_db")).get_collection("langchain")
    emb = get_embeddings()
    points, vectors = [], []
    for topic, page, question in TOPICS:
        got = col.get(where={"page": page - 1}, include=["embeddings", "documents"], limit=1)
        doc_vec, doc_text = got["embeddings"][0], got["documents"][0]
        q_vec = emb.embed_query(question)
        vectors += [unit(doc_vec), unit(q_vec)]
        snippet = " ".join(doc_text.split())[:40]
        points += [
            {"topic": topic, "kind": "매뉴얼 조각", "label": f"{page}쪽 · {snippet}…"},
            {"topic": topic, "kind": "질문", "label": question},
        ]

    X = np.vstack(vectors)
    sims = X @ X.T  # 코사인 유사도 (단위 벡터라 내적 = 코사인)
    for i in range(0, len(points), 2):  # 질문이 자기 주제 조각과 가장 가까운지 확인
        q = i + 1
        doc_scores = sims[q, 0::2]
        points[q]["nearest_topic"] = TOPICS[int(np.argmax(doc_scores))][0]
        points[q]["similarity_to_own"] = round(float(sims[q, i]), 3)

    matrix = [[round(float(sims[q, d]), 3) for d in range(0, len(points), 2)] for q in range(1, len(points), 2)]
    data = {
        "topics": [t for t, _, _ in TOPICS],
        "questions": [q for _, _, q in TOPICS],
        "chunks": [points[i]["label"] for i in range(0, len(points), 2)],
        "similarity": matrix,  # similarity[질문 i][조각 j]
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for p in points:
        if p["kind"] == "질문":
            ok = "O" if p["nearest_topic"] == p["topic"] else "X"
            print(f"{ok} {p['label']} → 가장 가까운 조각: {p['nearest_topic']} (자기 주제와 유사도 {p['similarity_to_own']})")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
