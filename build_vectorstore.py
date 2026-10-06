"""
매뉴얼 PDF를 Gemini로 임베딩해 chroma_db/ 에 저장하는 1회성 스크립트.

Gemini 무료 티어는 임베딩 요청이 분당 100회로 제한되어 있어
앱 시작 시마다 1,000개 넘는 청크를 임베딩하면 429 오류가 납니다.
그래서 로컬에서 이 스크립트로 한 번만 천천히 임베딩하고,
결과(chroma_db/)를 커밋해 앱에서는 불러오기만 합니다.

실행: .venv\\Scripts\\python build_vectorstore.py
중간에 한도(429)로 멈추면 다시 실행하면 이어서 진행합니다.
PDF나 분할 설정을 바꿨다면 chroma_db/ 폴더를 지우고 처음부터 다시 실행하세요.
(청크 ID가 순번이라 그대로 이어서 실행하면 이전 내용과 섞입니다.)
"""
import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from errors import classify, quota_ids
from vectorstore_config import PDF_PATH, PERSIST_DIR, get_embeddings

BATCH_SIZE = 20          # 한 번에 임베딩할 청크 수
REQUESTS_PER_MINUTE = 80  # 무료 티어 100회/분보다 여유 있게
MAX_RETRIES = 5           # 분당 한도 재시도 횟수 (넘으면 일일 한도로 보고 중단)


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    print("PDF 로딩 및 분할 중...")
    docs = PyPDFLoader(PDF_PATH).load()
    splits = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    ).split_documents(docs)
    ids = [f"chunk-{i:05d}" for i in range(len(splits))]

    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
    done = set(vectorstore.get(include=[])["ids"])
    todo = [(i, d) for i, d in zip(ids, splits) if i not in done]
    print(f"전체 {len(splits)}개 청크 중 {len(done)}개 완료, {len(todo)}개 남음")

    delay = 60 * BATCH_SIZE / REQUESTS_PER_MINUTE
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        for attempt in range(MAX_RETRIES + 1):
            try:
                vectorstore.add_documents([d for _, d in batch], ids=[i for i, _ in batch])
                break
            except Exception as e:
                kind = classify(e)
                if kind not in ("quota_day", "quota_minute", "unavailable"):
                    raise  # 한도·일시 장애가 아닌 오류는 그대로 보여줌
                if kind == "quota_day" or attempt == MAX_RETRIES:
                    print(f"  오류: {quota_ids(e) or str(e)[:200]}")
                    raise SystemExit("일일 무료 한도에 도달했습니다. 내일 다시 실행하면 이어서 진행합니다.")
                print("  분당 한도 초과/일시 장애 - 60초 대기 후 재시도")
                time.sleep(60)
        finished = len(done) + start + len(batch)
        print(f"  {finished}/{len(splits)} 완료")
        time.sleep(delay)

    print(f"완료! {PERSIST_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
