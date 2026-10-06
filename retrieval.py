"""
매뉴얼 검색 모듈.

- 기본: Gemini 임베딩으로 벡터 검색 (chroma_db/)
- 대체: 임베딩 API가 한도 초과(429) 등으로 실패하면 키워드 검색(BM25)으로 자동 전환
  키워드 검색은 API를 쓰지 않으므로 무료 한도가 바닥나도 데모가 멈추지 않습니다.

한국어는 띄어쓰기·조사 때문에 단어 단위 매칭이 약하므로 '글자 2개 묶음(bigram)'으로 비교합니다.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectorstore_config import PERSIST_DIR, get_embeddings


@dataclass
class SearchResult:
    docs: list[Document]
    method: str  # "vector" 또는 "keyword"


def _bigrams(text: str) -> list[str]:
    s = re.sub(r"\s+", "", text.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class KeywordIndex:
    """글자 bigram 기반 BM25 검색 (API 호출 없음)."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs = [Counter(_bigrams(d.page_content)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[Document]:
        terms = set(_bigrams(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
            for t in terms:
                f = tf.get(t)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(score)
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [self.docs[i] for i in top if scores[i] > 0]


class ManualRetriever:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        self.keyword = KeywordIndex(
            [Document(page_content=t, metadata=m or {}) for t, m in zip(raw["documents"], raw["metadatas"])]
        )

    def search(self, query: str, k: int = 3, force_keyword: bool = False) -> SearchResult:
        if not force_keyword:
            try:
                return SearchResult(self.vectorstore.similarity_search(query, k=k), "vector")
            except Exception as e:
                # 한도 초과(429)·일시 장애일 때만 키워드 검색으로 전환, 그 외 오류는 그대로 올림
                if not is_quota_or_unavailable(e):
                    raise
        return SearchResult(self.keyword.search(query, k=k), "keyword")


def is_quota_or_unavailable(e: Exception) -> bool:
    msg = str(e)
    return any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def page_label(doc: Document) -> int | None:
    """PDF 페이지 번호(1부터). metadata['page']는 0부터 시작합니다."""
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else None
