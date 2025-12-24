from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool

# .env 로드 및 검증
# .env 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("경고: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다. 시스템 환경 변수를 확인하세요.")
    if not os.getenv("OPENAI_API_KEY"):
         raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

app = FastAPI()

# Mount static files
app.mount("/img", StaticFiles(directory="img"), name="img")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에만 사용, 프로덕션에서는 특정 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 임베딩 & 벡터DB 초기화 (In-Memory) ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PDF 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Genesis_2026.pdf")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"매뉴얼 파일이 없습니다: {file_path}")

print("매뉴얼 로딩 및 임베딩 중... (서버 시작 시 약 10~20초 소요됩니다)")
loader = PyPDFLoader(file_path)
docs = loader.load()

# 2. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(docs)

# 3. 임베딩 모델
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)

# 4. 벡터 저장소 생성 (In-Memory)
# persist_directory를 지정하지 않으면 메모리에만 저장됨
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
print("매뉴얼 임베딩 완료!")

# === 검색 Tool 정의 ===
@tool(response_format="content_and_artifact")
def search_manual(query: str):
    """제네시스 차량 매뉴얼을 검색합니다. 차량 문제, 기능 사용법, 유지보수 정보 등을 찾을 때 사용하세요."""
    # Retriever로 검색
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    
    # 검색된 문서를 문자열로 포맷팅
    if not retrieved_docs:
        return "관련 정보를 찾을 수 없습니다.", []
    
    serialized = "\n\n".join(
        f"[페이지 {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    
    # content와 artifact(원본 문서) 모두 반환
    return serialized, retrieved_docs

# === LLM + Agent 설정 ===
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)

tools = [search_manual]

prompt = (
    "당신은 현대자동차 제네시스 매뉴얼 전문가입니다.\n"
    "사용자의 질문에 친절하고 전문적으로 답변해주세요.\n"
    "특히 안전과 관련된 내용은 반드시 강조해서 설명해주세요.\n\n"
    "매뉴얼을 검색할 때는 search_manual 도구를 사용하세요."
)

agent = create_agent(model, tools, system_prompt=prompt)

# === 요청 모델 ===
class Question(BaseModel):
    message: str

# === 헬스체크 엔드포인트 ===
@app.get("/")
async def root():
    return FileResponse("index.html")

# === 메시지 처리 API ===
@app.post("/chat")
async def chat(request: Question):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")
    
    user_msg = request.message.strip()
    
    try:
        # 에이전트 실행
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_msg}]
        })
        
        # 최종 답변 출력
        final_message = result["messages"][-1]
        return {"answer": final_message.content}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
