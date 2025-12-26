import streamlit as st
import streamlit.components.v1 as components
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ChatMessage, HumanMessage, AIMessage, SystemMessage

# === 설정 및 초기화 ===
st.set_page_config(page_title="제네시스 매뉴얼 챗봇", page_icon="🚗")
# === UI 스타일 적용 (Index.html 참고) ===
st.markdown("""
<style>
    /* 폰트 적용 */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans KR', sans-serif;
    }

    /* 메인 배경 설정 */
    .stApp {
        background-color: #0a0a0a;
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('https://www.genesis.com/content/dam/genesis-p2/kr/assets/main/hero/genesis-kr-main-kv-g90-lwb-black-main-hero-desktop-2560x900.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 모바일 반응형 배경 (index.html 참고) */
    @media (max-width: 768px) {
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('https://www.genesis.com/content/dam/genesis-p2/kr/assets/main/hero/genesis-kr-main-kv-g90-lwb-black-main-hero-mobile-750x1400.jpg');
        }
    }

    /* 헤더 텍스트 */
    h1 {
        background: linear-gradient(to right, #fff, #a38b6d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 300 !important;
    }
    
    /* Streamlit 상단 헤더 (Deploy 버튼 있는 영역) 투명화 */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        /* 또는 완전히 숨기기: display: none !important; */
    }
    
    /* 일반 텍스트 색상 */
    p, li, div {
        color: #e5e5e5;
    }

    /* Streamlit 채팅 입력창 커스터마이징 */
    /* 1. 입력창 전체 컨테이너 배경 투명화 */
    .stChatInput {
        background-color: transparent !important;
    }

    /* 2. 입력 상자 래퍼(둥근 모서리 요소)를 블랙으로 설정, 높이 증가 */
    div[data-testid="stChatInput"] > div {
        background-color: #1e1e1e !important;
        border-radius: 20px !important;
        border: 1px solid #333 !important;
        min-height: 60px !important; /* 높이 증가 */
        align-items: center !important; /* 수직 중앙 정렬 */
    }

    /* 2.5. 내부의 모든 div(배경색 가진 요소들) 투명화 - 흰색 배경 제거 핵심 */
    div[data-testid="stChatInput"] > div div {
        background-color: transparent !important;
    }

    /* 3. 내부 Textarea는 투명하게 (부모 배경색 사용) */
    .stChatInput textarea {
        background-color: transparent !important;
        color: #e5e5e5 !important;
        border: none !important;
        min-height: 40px !important; /* 높이 확보 */
        padding-right: 60px !important; /* 마이크 버튼 공간 확보 */
    }
    
    /* 4. 전송 버튼(비행기) 스타일 강화 */
    .stChatInput button {
        width: 45px !important;
        height: 45px !important;
        border: none !important;
        background: transparent !important;
        align-self: center !important; /* 수직 중앙 정렬 강제 */
        margin-top: auto !important;
        margin-bottom: auto !important;
    }
    .stChatInput button svg {
        width: 30px !important;
        height: 30px !important;
    }

    /* 5. 포커스 시 부모 래퍼 강조 */
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #a38b6d !important;
        box-shadow: 0 0 0 1px #a38b6d !important;
    }

    /* Placeholder 색상 변경 */
    .stChatInput textarea::placeholder {
        color: #e5e5e5 !important;
        opacity: 0.8;
    }

    /* 6. 플로팅 토글 버튼 (메인 및 전역) */
    div[data-testid="stButton"], div.stButton {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 9999 !important;
        width: 50px !important;
        height: 50px !important;
    }
    div[data-testid="stButton"] > button, div.stButton > button {
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        background-color: #a38b6d !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 32px !important;
        line-height: 1 !important;
    }
    div[data-testid="stButton"] > button:hover, div.stButton > button:hover {
        transform: scale(1.1);
        background-color: #b59c7d !important;
    }

    /* 사이드바 버튼 원래대로 복구 (Global !important 덮어쓰기) */
    [data-testid="stSidebar"] div[data-testid="stButton"] {
        position: static !important;
        width: auto !important;
        height: auto !important;
        margin-top: 10px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100% !important;
        height: auto !important;
        border-radius: 8px !important; /* 모서리 살짝 둥글게 */
        background-color: #262730 !important;
        color: white !important;
        box-shadow: none !important;
        padding: 0.5rem 1rem !important;
        display: inline-flex !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        transform: none !important;
        background-color: #3e404b !important;
    }
    /* Mobile Responsiveness */
    @media only screen and (max-width: 600px) {
        h1, h1 span {
            font-size: 24px !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h1 span {
            font-size: 20px !important;
        }
        /* 본문 및 채팅 폰트 축소 */
        html, body, p, li, div, span, button, [class*="css"] {
            font-size: 14px !important;
        }
        /* 특정 컴포넌트 예외 처리 (필요시) */
        .stMarkdown p {
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
        /* 버튼 텍스트도 줄임 (아이콘 제외) */
        button p {
            font-size: 14px !important;
        }
    }

    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
        border-right: 1px solid #333 !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #fff !important;
        font-weight: 300 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] span {
        color: #e5e5e5 !important;
    }
    /* 탭 스타일링 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #a38b6d !important; /* 골드 컬러 */
        border-bottom-color: #a38b6d !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("GENESIS AI Assistant")

# 1. API Key 설정 (Streamlit Secrets 우선, 없으면 로컬 .env)


try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    # 로컬 환경 등 secrets가 없는 경우 무시
    pass

# 환경 변수가 없으면 .env 로드 시도
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 .env 파일을 확인해주세요.")
    st.stop()

# 2. 리소스 캐싱 (PDF 로드 및 벡터 DB 생성은 한 번만 실행)
@st.cache_resource
def initialize_vector_store():
    with st.spinner("매뉴얼을 로딩하고 분석 중입니다... (최초 1회만 실행됨)"):
        # PDF 파일 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "Genesis_2026.pdf")
        
        if not os.path.exists(file_path):
            st.error(f"매뉴얼 파일이 없습니다: {file_path}")
            st.stop()

        # PDF 로드
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(docs)

        # 임베딩 모델
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
        )

        # 인메모리 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        return vectorstore

# 벡터 스토어 초기화
vectorstore = initialize_vector_store()

# 3. 에이전트 도구 및 모델 설정
@tool
def search_manual(query: str):
    """제네시스 차량 매뉴얼을 검색합니다. 차량 문제, 기능 사용법, 유지보수 정보 등을 찾을 때 사용하세요."""
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    
    if not retrieved_docs:
        return "관련 정보를 찾을 수 없습니다."
    
    serialized = "\n\n".join(
        f"[페이지 {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized

# LLM & Agent 설정
# Chat History 변환 헬퍼 함수
def get_chat_history(messages):
    history = []
    # 마지막 메시지는 'input'이므로 제외
    for msg in messages[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# LLM & Agent 설정
if "agent" not in st.session_state:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    tools = [search_manual]
    
    system_prompt = (
        "당신은 현대자동차 제네시스 매뉴얼 전문가입니다.\n"
        "사용자의 질문에 친절하고 전문적으로 답변해주세요.\n"
        "특히 안전과 관련된 내용은 반드시 강조해서 설명해주세요.\n"
        "매뉴얼을 검색할 때는 search_manual 도구를 사용하세요."
    )

    # Agent 생성 (Custom create_agent 사용)
    # create_agent는 CompiledStateGraph를 반환하며, 이는 Runnable입니다.
    st.session_state.agent = create_agent(model, tools, system_prompt=system_prompt)

# 4. 채팅 UI 및 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 제네시스 차량에 대해 궁금한 점을 물어보세요."}
    ]

# 채팅창 표시 여부 상태 관리
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# === 사이드바 (설정 및 도움말) ===
with st.sidebar:
    st.title("GENESIS Assistant")
    
    # 탭 분리
    tab1, tab2 = st.tabs(["가이드 💡", "대화 관리 ⚙️"])
    
    with tab1:
        st.subheader("사용법")
        st.markdown("""
        1. **우측 하단 아이콘**을 눌러 대화를 시작하세요.
        2. **차량 기능, 유지보수, 문제 해결**에 대해 물어보세요.
        3. 예시:
            - "타이어 공기압은 얼마나 넣어야 해?"
            - "스마트 키 배터리 교체 방법 알려줘"
            - "엔전 오일 경고등이 떴어"
        """)
        
    with tab2:
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 내용 지우기", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "안녕하세요! 제네시스 차량에 대해 궁금한 점을 물어보세요."}
            ]
            st.rerun()

    st.markdown("---")


# 토글 버튼 (우측 하단)
def toggle_chat():
    st.session_state.show_chat = not st.session_state.show_chat

# 채팅방이 열려있으면 X(닫기), 닫혀있으면 💬(열기) 표시
toggle_icon = "✖" if st.session_state.get("show_chat", False) else "💬"
st.button(toggle_icon, on_click=toggle_chat, key="toggle_chat_btn_v4")

# 채팅창이 활성화된 경우에만 표시
if st.session_state.show_chat:
    # 이전 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 입력창 표시
    prompt = st.chat_input("질문을 입력하세요 (예: 타이어 공기압은?)")
else:
    prompt = None
    # 대기 화면 안내 (선택)
    st.markdown(
        """
        <div style='position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: rgba(255,255,255,0.7); pointer-events: none;'>
            <h1 style='font-weight: 300;'>GENESIS AI</h1>
            <p>우측 하단 아이콘을 눌러 대화를 시작하세요</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 사용자 입력 처리
if prompt:
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("매뉴얼 검색 중..."):
            try:
                # History 변환 (Dict -> BaseMessage)
                # create_agent (LangGraph 기반)는 messages 리스트를 받아 처리합니다.
                chat_history = []
                for msg in st.session_state.messages:
                   if msg["role"] == "user":
                       chat_history.append(HumanMessage(content=msg["content"]))
                   elif msg["role"] == "assistant":
                       chat_history.append(AIMessage(content=msg["content"]))

                # invoke 호출 (전체 히스토리 전달)
                response = st.session_state.agent.invoke({
                    "messages": chat_history
                })
                # LangGraph response는 dict이며 'messages' 키에 전체 대화가 들어있고, 마지막이 답변입니다.
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# === 음성 인식 컴포넌트 (Javascript Injection via iframe) ===
# 부모 창의 DOM을 직접 조작하여 플로팅 버튼과 오버레이를 주입합니다.
# 이렇게 하면 iframe의 크기 제약 없이 전체 화면 오버레이를 구현할 수 있습니다.

js_code = """
<script>
    (function() {
        const parentDoc = window.parent.document;
        
        // === 기존 요소 제거 (재실행 시 핸들러 갱신의 핵심) ===
        // Streamlit이 다시 실행될 때마다 새로운 iframe이 생성되는데, 
        // 기존 버튼이 남아있으면 이전 iframe 컨텍스트의 핸들러(이미 죽은 객체)를 참조하게 됩니다.
        // 따라서 기존 버튼을 제거하고 새로 만들어야 합니다.
        const elementIds = ["voice-trigger-btn", "voice-overlay", "voice-custom-style"];
        elementIds.forEach(id => {
            const el = parentDoc.getElementById(id);
            if (el) el.remove();
        });

        // 1. CSS 스타일 주입
        const style = parentDoc.createElement("style");
        style.id = "voice-custom-style";
        style.innerHTML = `
            #voice-trigger-btn {
                position: fixed;
                bottom: 100px;
                right: 30px;
                width: 50px;
                height: 50px;
                background-color: #a38b6d;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                z-index: 999999;
                transition: transform 0.2s, background-color 0.2s;
            }
            #voice-trigger-btn:hover {
                transform: scale(1.1);
                background-color: #b59c7d;
            }
            #voice-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(10, 10, 10, 0.9);
                z-index: 1000000;
                display: none;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 20px;
                backdrop-filter: blur(5px);
            }
            .voice-status {
                color: #e5e5e5;
                font-size: 1.5rem;
                font-weight: 300;
            }
            .mic-ring {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                border: 2px solid #a38b6d;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                color: #a38b6d;
            }
            .mic-ring.active {
                animation: pulse 1.5s infinite;
                background-color: rgba(163, 139, 109, 0.2);
            }
            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(163, 139, 109, 0.4); }
                70% { transform: scale(1.1); box-shadow: 0 0 0 20px rgba(163, 139, 109, 0); }
                100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(163, 139, 109, 0); }
            }
        `;
        parentDoc.head.appendChild(style);

        // 2. HTML 요소 생성 (버튼)
        const btn = parentDoc.createElement("div");
        btn.id = "voice-trigger-btn";
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
            </svg>
        `;
        parentDoc.body.appendChild(btn);

        // 3. HTML 요소 생성 (오버레이)
        const overlay = parentDoc.createElement("div");
        overlay.id = "voice-overlay";
        overlay.innerHTML = `
            <div class="voice-status" id="v-status">듣는 중...</div>
            <div class="mic-ring" id="v-ring">🎤</div>
            <button id="v-cancel" style="margin-top:20px; padding:8px 20px; border-radius:15px; border:1px solid #666; background:transparent; color:#ccc; cursor:pointer;">취소</button>
        `;
        parentDoc.body.appendChild(overlay);

        // 4. 로직 구현
        let recognition = null;
        
        // 브라우저 호환성 및 HTTPS 확인
        if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            console.warn("음성 인식은 HTTPS 또는 로컬 환경에서만 작동합니다.");
        }

        var SpeechRecognition = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.lang = 'ko-KR';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = function() {
                overlay.style.display = 'flex';
                parentDoc.getElementById("v-status").innerText = "말씀하세요...";
                parentDoc.getElementById("v-ring").classList.add("active");
            };

            recognition.onend = function() {
                overlay.style.display = 'none';
                parentDoc.getElementById("v-ring").classList.remove("active");
            };

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                
                // 1. 우선적으로 data-testid로 시도
                let chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                
                // 2. 실패시, 모든 textarea 중 마지막 요소 선택
                if (!chatInput) {
                    const allTextAreas = parentDoc.getElementsByTagName('textarea');
                    if (allTextAreas.length > 0) {
                        chatInput = allTextAreas[allTextAreas.length - 1];
                    }
                }

                if (chatInput) {
                    // React 상태 업데이트를 위해 네이티브 value setter 사용
                    const nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
                    nativeTextAreaValueSetter.call(chatInput, transcript);
                    
                    // Input 이벤트 발생
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // 잠시 대기 후 Enter키 전송
                    setTimeout(() => {
                        chatInput.focus();
                        const enterEvent = new KeyboardEvent('keydown', {
                            bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13
                        });
                        chatInput.dispatchEvent(enterEvent);
                    }, 100);
                } else {
                    console.error("No textarea found in parent document.");
                    alert("입력창을 찾을 수 없습니다. (Textarea 요소 없음)");
                }
            };
            
            recognition.onerror = function(event) {
                console.error("Speech recognition error", event.error);
                if (event.error === 'not-allowed') {
                    alert("마이크 사용 권한이 차단되었습니다. 브라우저 설정에서 허용해주세요.");
                } else {
                    // 기타 오류는 조용히 로그만 남김 (사용자 방해 최소화)
                }
                overlay.style.display = 'none';
            };
        } else {
             console.warn("이 브라우저는 음성 인식을 지원하지 않습니다.");
        }

        function startVoiceRecognition() {
            if (!recognition) return;
            window.parent.navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    recognition.start();
                })
                .catch(function(err) {
                    alert("마이크 권한 오류: " + err.name + "\\n브라우저 설정에서 마이크를 허용해주세요.\\n(주의: localhost 또는 HTTPS 환경이어야 합니다.)");
                });
        }

        // 이벤트 리스너 연결
        btn.onclick = function() {
            if (!recognition) {
                alert("이 브라우저는 음성 인식을 지원하지 않습니다.");
                return;
            }
            
            // 1. 채팅창이 이미 열려있는지 확인
            let chatInput = parentDoc.querySelector('textarea');
            
            if (chatInput) {
                startVoiceRecognition();
            } else {
                // 2. 닫혀있다면 토글 버튼 클릭 (페이지 리로드 유발)
                const buttons = Array.from(parentDoc.querySelectorAll('button'));
                const toggleBtn = buttons.find(b => b.innerText.includes('💬'));
                
                if (toggleBtn) {
                    // 리로드 후 자동 실행을 위해 sessionStorage에 플래그 저장
                    // 중요: 리로드 후에는 버튼 클릭 없이 실행되므로 '사용자 제스처' 이슈가 있을 수 있음.
                    // 이를 위해 사용자에게 명확한 피드백을 주는 것이 좋음.
                    window.parent.sessionStorage.setItem("auto_start_voice", "true");
                    toggleBtn.click();
                } else {
                    alert("대화창을 자동으로 열 수 없습니다.");
                }
            }
        };
        
        parentDoc.getElementById("v-cancel").onclick = function() {
            if (recognition) recognition.stop();
            overlay.style.display = 'none';
        }

        // === 페이지 리로드 후 자동 실행 체크 ===
        if (window.parent.sessionStorage.getItem("auto_start_voice") === "true") {
            window.parent.sessionStorage.removeItem("auto_start_voice");
            
            // 1. 시각적 피드백 즉시 제공
            overlay.style.display = 'flex';
            parentDoc.getElementById("v-status").innerText = "대화창 준비 중...";
            
            // 2. 안정적인 실행을 위해 1초 대기 (Streamlit 렌더링 완료 확보)
            setTimeout(() => {
                parentDoc.getElementById("v-status").innerText = "음성 인식을 시작합니다...";
                startVoiceRecognition(true); // isAutoStart = true
            }, 1000);
        }

        function startVoiceRecognition(isAutoStart = false) {
            if (!recognition) return;
            window.parent.navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    recognition.start();
                })
                .catch(function(err) {
                    overlay.style.display = 'none'; // 오류 시 오버레이 숨김
                    
                    // 브라우저 자동 재생 정책 등으로 막혔을 경우
                    if (isAutoStart) {
                        alert("대화창이 열렸습니다! 마이크 버튼을 한 번 더 눌러 말씀을 시작해 주세요. (브라우저 보안)");
                    } else {
                        alert("마이크 권한 오류: " + err.name + "\\n브라우저 설정에서 마이크를 허용해주세요.");
                    }
                });
        }
    })();
</script>
"""
components.html(js_code, height=0)
