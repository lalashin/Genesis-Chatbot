# 필수 라이브러리 임포트
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# .env 파일 로드
load_dotenv()

# OpenAI 임베딩 모델 초기화 (벡터 저장소와 동일한 설정 필요)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)

# 기존 벡터 저장소 로드
print("\n=== 벡터 저장소 로드 중 ===")
vectorstore = Chroma(
    persist_directory="./.chroma_db",  # Genesis_1.py에서 생성한 저장소 경로
    embedding_function=embeddings
)

# 검색기(Retriever) 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 결과 검색

# 검색기 테스트
query = "타이어가 펑크났어. 해결책을 알려줘"
print(f"\n=== 기본 검색 테스트 ===")# 필수 라이브러리 임포트
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# .env 파일 로드
load_dotenv()

# OpenAI 임베딩 모델 초기화 (벡터 저장소와 동일한 설정 필요)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)

# 기존 벡터 저장소 로드
print("\n=== 벡터 저장소 로드 중 ===")
vectorstore = Chroma(
    persist_directory="./.chroma_db",  # Genesis_1.py에서 생성한 저장소 경로
    embedding_function=embeddings
)

# 검색기(Retriever) 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 결과 검색

# 추가 라이브러리 임포트
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Retriever를 Tool로 변환 (공식 방식)
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

# LLM 모델 초기화
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2
)

# 에이전트 생성
tools = [search_manual]
prompt = (
    "당신은 현대자동차 제네시스 매뉴얼 전문가입니다.\n"
    "사용자의 질문에 친절하고 전문적으로 답변해주세요.\n"
    "특히 안전과 관련된 내용은 반드시 강조해서 설명해주세요.\n\n"
    "매뉴얼을 검색할 때는 search_manual 도구를 사용하세요."
)

agent = create_agent(model, tools, system_prompt=prompt)

# Q&A 대화형 인터페이스
print("\n=== 제네시스 매뉴얼 Q&A 챗봇 ===")
print("종료하려면 'q' 또는 'quit'를 입력하세요.\n")

while True:
    user_question = input("질문: ")
    
    if user_question.lower() in ['q', 'quit', '종료']:
        print("\n챗봇을 종료합니다.")
        break
    
    if not user_question.strip():
        continue
    
    try:
        # 에이전트 실행
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_question}]
        })
        
        # 최종 답변 출력
        final_message = result["messages"][-1]
        print(f"\n답변: {final_message.content}\n")
        print("-" * 70 + "\n")
        
    except Exception as e:
        print(f"\n오류 발생: {e}\n")
        continue
print(f"검색어: {query}")

results = vectorstore.similarity_search(query, k=3)
print("\n=== 검색 결과 ===")
for i, doc in enumerate(results, 1):
    print(f"\n[검색 결과 {i}]")
    print(f"페이지: {doc.metadata.get('page', 'N/A')}")
    print(f"내용: {doc.page_content}\n")