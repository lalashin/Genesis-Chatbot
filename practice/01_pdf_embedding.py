# PDF 문서 로드
# PDF 파일 경로 설정
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(current_dir, "Genesis_2026.pdf")

# PyPDF 문서 로드
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"\n=== 일반 로드 결과 ===")
print(f"문서 페이지 수: {len(docs)}")

# PDF 문서 분할 
# RecursiveCharacterTextSplitter 문서 분할
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],  # 분할 구분자 우선순위
    chunk_size=1000,  # 청크 크기 (글자 수)
    chunk_overlap=200,  # 청크 간 중복 영역 (글자 수)
    length_function=len  # 길이 계산 함수
)

splits = text_splitter.split_documents(docs)
print(f"\n=== 문서 분할 결과 ===")
print(f"분할된 청크 수: {len(splits)}")
print(f"\n=== 첫 번째 청크 내용 예시 ===")
print(splits[0].page_content)

# OpenAI 임베딩
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드
from langchain_openai import OpenAIEmbeddings  # OpenAI 임베딩 모듈
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 임베딩 모델 선택
    dimensions=1536,  # 벡터 개수 설정
)

print(f"\n=== 임베딩 테스트 결과 ===")
test_vector = embeddings.embed_query(splits[0].page_content)
print(f"임베딩 벡터 차원 수: {len(test_vector)}")
print(f"임베딩 벡터 예시 (앞 5개): {test_vector[:5]}...")

# Chroma 저장소 생성
from langchain_community.vectorstores import Chroma  # Chroma 벡터 저장소
vectorstore = Chroma.from_documents(
    documents=splits,  # 분할된 문서 목록
    embedding=embeddings,  # 임베딩 모델
    persist_directory="./.chroma_db"  # 벡터 저장소 저장 경로
)

print("\n=== 벡터 저장소 생성 완료 ===")
print(f"저장 위치: {os.path.abspath('./.chroma_db')}")