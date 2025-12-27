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



# === [신규 기능] 음성 비서 온보딩 (Toggle Switch) ===
if "voice_onboarded" not in st.session_state:
    st.session_state.voice_onboarded = False

# [UI 개선] 토글 스위치 크기 확대 (PC/Mobile 공통)
st.markdown("""
<style>
    /* 토글 스위치 컨테이너 전체 확대 */
    div[data-testid="stToggle"] label {
        font-size: 20px !important; /* 라벨 폰트 키움 */
        align-items: center !important;
    }
    /* 토글 스위치 본체 확대 (Checkbox input + span) */
    div[data-testid="stToggle"] p, div[data-testid="stToggle"] span {
        font-weight: 600 !important;
    }
    /* 실제 스위치 부분 확대 */
    div[data-testid="stCheckbox"] {
        transform: scale(1.5) !important; /* 체크박스(스위치) 1.5배 확대 */
        margin-right: 15px !important; /* 텍스트와의 간격 조정 */
        transform-origin: left center !important;
    }
</style>
""", unsafe_allow_html=True)

# Toggle Switch (Visual: "음성 인식 활성화")
# 사용자 요청: "GENESIS AI Assistant" 바로 위에 배치
toggle_label = "음성 AI 비서가 활성화 되었습니다." if st.session_state.voice_onboarded else "음성 AI 비서를 활성화 해주세요!"
on_toggle = st.toggle(toggle_label, value=st.session_state.voice_onboarded)

if on_toggle:
    if not st.session_state.voice_onboarded:
        st.session_state.voice_onboarded = True
        st.rerun()
    pass

elif not on_toggle and st.session_state.voice_onboarded:
    st.session_state.voice_onboarded = False
    st.rerun()

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
            separators=["\\n\\n", "\\n", ".", " "],
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
    
    serialized = "\\n\\n".join(
        f"[페이지 {doc.metadata.get('page', 'N/A')}]\\n{doc.page_content}"
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
        "당신은 현대자동차 제네시스 매뉴얼 전문가입니다.\\n"
        "사용자의 질문에 친절하고 전문적으로 답변해주세요.\\n"
        "특히 안전과 관련된 내용은 반드시 강조해서 설명해주세요.\\n"
        "매뉴얼을 검색할 때는 search_manual 도구를 사용하세요."
    )

    # Agent 생성 (Custom create_agent 사용)
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

js_code = """
<script>
    (function() {
        const parentDoc = window.parent.document;
        const btnId = "voice-trigger-btn";
        const overlayId = "voice-overlay";
        const tooltipId = "voice-tooltip";
        const styleId = "voice-custom-style";
        
        // [State Injection]
        const isOnboarded = IS_ONBOARDED_PLACEHOLDER;

        // 1. CSS Injection (Idempotent)
        if (!parentDoc.getElementById(styleId)) {
            const style = parentDoc.createElement("style");
            style.id = styleId;
            style.innerHTML = `
                #voice-trigger-btn {
                    position: fixed; bottom: 100px; right: 30px; width: 50px; height: 50px;
                    background-color: #a38b6d; border-radius: 50%; display: flex;
                    align-items: center; justify-content: center; cursor: pointer;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.3); z-index: 999999;
                    transition: transform 0.2s, background-color 0.2s;
                }
                #voice-trigger-btn:hover { transform: scale(1.1); background-color: #b59c7d; }
                #voice-overlay {
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background-color: rgba(10, 10, 10, 0.9); z-index: 1000000;
                    display: none; flex-direction: column; align-items: center; justify-content: center;
                    gap: 20px; backdrop-filter: blur(5px);
                }
                .voice-status { color: #e5e5e5; font-size: 1.5rem; font-weight: 300; }
                .mic-ring {
                    width: 80px; height: 80px; border-radius: 50%; border: 2px solid #a38b6d;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 2rem; color: #a38b6d;
                }
                .mic-ring.active { animation: pulse 1.5s infinite; background-color: rgba(163, 139, 109, 0.2); }
                @keyframes pulse {
                    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(163, 139, 109, 0.4); }
                    70% { transform: scale(1.1); box-shadow: 0 0 0 20px rgba(163, 139, 109, 0); }
                    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(163, 139, 109, 0); }
                }
                #voice-tooltip {
                    position: fixed; bottom: 160px; right: 25px; background-color: #333; color: #fff;
                    padding: 10px 15px; border-radius: 8px; font-size: 14px; font-weight: 500;
                    white-space: nowrap; z-index: 999999; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    pointer-events: none; display: none; opacity: 0; transition: opacity 0.3s;
                }
                #voice-tooltip.visible { display: block; opacity: 1; }
                #voice-tooltip::after {
                    content: ''; position: absolute; top: 100%; left: 75%; margin-left: -6px;
                    border-width: 6px; border-style: solid; border-color: #333 transparent transparent transparent;
                }
            `;
            parentDoc.head.appendChild(style);
        }

        // 2. DOM Elements (Ensure Existence)
        let btn = parentDoc.getElementById(btnId);
        if (!btn) {
            btn = parentDoc.createElement("div");
            btn.id = btnId;
            btn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <line x1="12" y1="19" x2="12" y2="23"></line>
                </svg>
            `;
            parentDoc.body.appendChild(btn);
        }

        let tooltip = parentDoc.getElementById(tooltipId);
        if (!tooltip) {
            tooltip = parentDoc.createElement("div");
            tooltip.id = tooltipId;
            tooltip.innerText = "음성 비서 활성화 하세요";
            parentDoc.body.appendChild(tooltip);
        }

        let overlay = parentDoc.getElementById(overlayId);
        if (!overlay) {
            overlay = parentDoc.createElement("div");
            overlay.id = overlayId;
            overlay.innerHTML = `
                <div class="voice-status" id="v-status">듣는 중...</div>
                <div class="mic-ring" id="v-ring">🎤</div>
                <button id="v-cancel" style="margin-top:20px; padding:8px 20px; border-radius:15px; border:1px solid #666; background:transparent; color:#ccc; cursor:pointer;">취소</button>
            `;
            parentDoc.body.appendChild(overlay);
        }

        // 3. Conditional UI Logic
        if (!isOnboarded) {
             if(tooltip) tooltip.classList.add("visible");
        } else {
             if(tooltip) tooltip.classList.remove("visible");
        }

        // 4. SpeechRecognition Component
        var SpeechRecognition = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
        
        // Helper: 인스턴스 생성 및 설정 함수
        function getOrCreateRecognition() {
            if (!SpeechRecognition) return null;
            if (!window.parent._voice_recog_instance) {
                const recog = new SpeechRecognition();
                recog.lang = 'ko-KR';
                recog.continuous = false;
                recog.interimResults = false;
                window.parent._voice_recog_instance = recog;
            }
            // 핸들러는 항상 갱신 (Closure 갱신) for PC Overlay Fix
            const recog = window.parent._voice_recog_instance;
            
            recog.onstart = function() {
                const ov = parentDoc.getElementById(overlayId);
                if(ov) ov.style.display = 'flex';
                const st = parentDoc.getElementById("v-status");
                if(st) st.innerText = "말씀하세요...";
                const ring = parentDoc.getElementById("v-ring");
                if(ring) ring.classList.add("active");
            };
            recog.onend = function() {
                const ov = parentDoc.getElementById(overlayId);
                if(ov) ov.style.display = 'none';
                const ring = parentDoc.getElementById("v-ring");
                if(ring) ring.classList.remove("active");
            };
            recog.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                let chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (!chatInput) {
                    const allTextAreas = parentDoc.getElementsByTagName('textarea');
                    if (allTextAreas.length > 0) chatInput = allTextAreas[allTextAreas.length - 1];
                }
                if (chatInput) {
                    const nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
                    nativeTextAreaValueSetter.call(chatInput, transcript);
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                    setTimeout(() => {
                        chatInput.focus();
                        const enterEvent = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13 });
                        chatInput.dispatchEvent(enterEvent);
                    }, 100);
                }
            };
            recog.onerror = function(event) {
                const ov = parentDoc.getElementById(overlayId);
                if(ov) ov.style.display = 'none';
                // 권한 거부 시 안내
                console.warn("Voice Error:", event.error);
                if (event.error === 'not-allowed') {
                    // 사용자 경험상 "권한 묻기" 단계에서 거절하면 다시 안 뜨는 게 나을 수도 있음
                }
            };
            return recog;
        }

        // 5. [NEW] Toggle Click Interceptor (토글 클릭 시 권한 선제 요청)
        const toggleContainer = parentDoc.querySelector('div[data-testid="stToggle"]');
        if (toggleContainer) {
            toggleContainer.onmousedown = function() {
                // 토글을 누르는 순간 -> 마이크 권한 요청 시도
                if (!window.parent._voice_recog_instance) {
                    const recog = getOrCreateRecognition();
                    if (recog) {
                        try {
                            recog.start();
                        } catch(e) { console.log("Priming error:", e); }
                    }
                }
            };
        }

        // 6. Mic Button Click Handler
        btn.onclick = function() {
            if (!isOnboarded) {
                const tt = parentDoc.getElementById(tooltipId);
                if(tt) {
                    tt.classList.add("visible");
                    setTimeout(() => tt.classList.remove("visible"), 2000);
                }
                return;
            }
            
            const recognition = getOrCreateRecognition();
            if (!recognition) {
                alert("음성 인식을 지원하지 않는 브라우저입니다.");
                return;
            }

            // 채팅창 확인 및 실행
            let chatInput = parentDoc.querySelector('textarea');
            if (chatInput) {
                try { recognition.start(); } catch(e) { console.error(e); }
            } else {
                 const buttons = Array.from(parentDoc.querySelectorAll('button'));
                 const toggleBtn = buttons.find(b => b.innerText.includes('💬'));
                 if (toggleBtn) {
                     window.parent.sessionStorage.setItem("auto_start_voice", "true");
                     toggleBtn.click();
                 }
            }
        };

        const cancelBtn = parentDoc.getElementById("v-cancel");
        if(cancelBtn) {
            cancelBtn.onclick = function() {
                const recognition = window.parent._voice_recog_instance;
                if (recognition) recognition.stop();
            };
        }
        
        // 자동 실행 체크 (페이지 로드 후)
        if (window.parent.sessionStorage.getItem("auto_start_voice") === "true") {
            window.parent.sessionStorage.removeItem("auto_start_voice");
            setTimeout(() => {
                const recognition = getOrCreateRecognition();
                 if(recognition) {
                    let chatInput = parentDoc.querySelector('textarea');
                    if (chatInput) try { recognition.start(); } catch(e) {}
                 }
            }, 1000);
        }

    })();
</script>
"""
js_code = js_code.replace("IS_ONBOARDED_PLACEHOLDER", str(st.session_state.voice_onboarded).lower())
components.html(js_code, height=0)