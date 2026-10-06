# 벡터 DB 공통 설정 (build_vectorstore.py, streamlit_app.py 에서 함께 사용)
# 임베딩 모델/차원을 바꾸면 build_vectorstore.py 로 chroma_db/ 를 다시 만들어야 합니다.
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Genesis_2026.pdf")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
    )
