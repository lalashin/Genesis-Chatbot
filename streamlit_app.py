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
from langchain_core.messages import ChatMessage

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
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://www.genesis.com/content/dam/genesis-p2/kr/assets/main/hero/genesis-kr-main-kv-g90-lwb-black-main-hero-desktop-2560x900.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 모바일 반응형 배경 (index.html 참고) */
    @media (max-width: 768px) {
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://www.genesis.com/content/dam/genesis-p2/kr/assets/main/hero/genesis-kr-main-kv-g90-lwb-black-main-hero-mobile-750x1400.jpg');
        }
    }

    /* 헤더 텍스트 */
    h1 {
        background: linear-gradient(to right, #fff, #a38b6d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 300 !important;
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
        background-color: #000000 !important;
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
if "agent" not in st.session_state:
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    tools = [search_manual]
    system_prompt = (
        "당신은 현대자동차 제네시스 매뉴얼 전문가입니다.\n"
        "사용자의 질문에 친절하고 전문적으로 답변해주세요.\n"
        "특히 안전과 관련된 내용은 반드시 강조해서 설명해주세요.\n"
        "매뉴얼을 검색할 때는 search_manual 도구를 사용하세요."
    )
    st.session_state.agent = create_agent(model, tools, system_prompt=system_prompt)

# 4. 채팅 UI 및 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 제네시스 차량에 대해 궁금한 점을 물어보세요."}
    ]

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요 (예: 타이어 공기압은?"):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("매뉴얼 검색 중..."):
            try:
                response = st.session_state.agent.invoke({
                    "messages": st.session_state.messages
                })
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# === 음성 인식 컴포넌트 (Javascript Injection via iframe) ===
# 부모 창(Streamlit 메인 UI)의 DOM을 직접 조작하여 플로팅 버튼과 오버레이를 주입합니다.
# 이렇게 하면 iframe의 크기 제약 없이 전체 화면 오버레이를 구현할 수 있습니다.

js_code = """
<script>
    (function() {
        const parentDoc = window.parent.document;
        
        // 이미 생성되었는지 확인
        if (parentDoc.getElementById("voice-trigger-btn")) {
            return;
        }

        // 1. CSS 스타일 주입
        const style = parentDoc.createElement("style");
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
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
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
                
                // Streamlit 입력창 찾기
                const chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (chatInput) {
                    const nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeTextAreaValueSetter.call(chatInput, transcript);
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // 자동 전송 (약간의 지연)
                    setTimeout(() => {
                        const enterEvent = new KeyboardEvent('keydown', {
                            bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13
                        });
                        chatInput.dispatchEvent(enterEvent);
                    }, 200);
                }
            };
        }

        // 이벤트 리스너 연결
        btn.onclick = function() {
            if (recognition) recognition.start();
            else alert("이 브라우저는 음성 인식을 지원하지 않습니다.");
        };
        
        parentDoc.getElementById("v-cancel").onclick = function() {
            if (recognition) recognition.stop();
            overlay.style.display = 'none';
        }

    })();
</script>
"""
components.html(js_code, height=0)
# 부모 창(Streamlit 메인 UI)의 DOM을 직접 조작하여 플로팅 버튼과 오버레이를 주입합니다.
# 이렇게 하면 iframe의 크기 제약 없이 전체 화면 오버레이를 구현할 수 있습니다.

js_code = """
<script>
    (function() {
        const parentDoc = window.parent.document;
        
        // 이미 생성되었는지 확인
        if (parentDoc.getElementById("voice-trigger-btn")) {
            return;
        }

        // 1. CSS 스타일 주입
        const style = parentDoc.createElement("style");
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
        // iframe 내에서는 마이크 권한 요청이 막힐 수 있으므로, 부모 창(Main App)의 webkitSpeechRecognition을 사용
        if ('webkitSpeechRecognition' in window.parent) {
            recognition = new window.parent.webkitSpeechRecognition();
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
                
                // Streamlit 입력창 찾기
                const chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (chatInput) {
                    const nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeTextAreaValueSetter.call(chatInput, transcript);
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // 자동 전송 (약간의 지연)
                    setTimeout(() => {
                        const enterEvent = new KeyboardEvent('keydown', {
                            bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13
                        });
                        chatInput.dispatchEvent(enterEvent);
                    }, 200);
                }
            };
        }

        // 이벤트 리스너 연결
        btn.onclick = function() {
            if (!recognition) {
                alert("이 브라우저는 음성 인식을 지원하지 않습니다.");
                return;
            }
            // 마이크 권한 명시적 요청
            window.parent.navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    recognition.start();
                })
                .catch(function(err) {
                    alert("마이크 권한 오류: " + err.name + "\n브라우저 설정에서 마이크를 허용해주세요.\n(주의: localhost 또는 HTTPS 환경이어야 합니다.)");
                });
        };
        
        parentDoc.getElementById("v-cancel").onclick = function() {
            if (recognition) recognition.stop();
            overlay.style.display = 'none';
        }

    })();
</script>
"""
components.html(js_code, height=0)
