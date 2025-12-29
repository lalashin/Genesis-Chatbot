배포링크: https://genesis-chatbot.streamlit.app/


feat(ui): Enhance UI/UX and implement conversation memory

- Implement session-based conversation memory
- Restructure sidebar with tabs for Guide and Conversation Management
- Refine dark theme colors and adjust brightness for better visibility
- Fix floating button CSS conflicts and restore sidebar button styling
- Optimize chat toggle button behavior

📅 작업 내역 요약 (Feature: UI/UX Improvements & Memory)
1. 대화 기억 기능 (Conversation Memory)

세션 기반의 대화 기억 로직 구현 (
get_chat_history
)
사용자 환경(LangChain 버전 이슈)에 맞춰 Agent 생성 로직 복구 및 최적화
2. 사이드바(Sidebar) 구조 개선

탭 분리: '가이드 💡'와 '대화 관리 ⚙️' 탭으로 기능을 명확히 분리
초기화 버튼: '대화 내용 지우기' 버튼을 '대화 관리' 탭 내부로 이동 및 재배치
스타일링: 사이드바 전용 버튼 스타일(사각형, radius 8px) 적용으로 가시성 개선
3. UI/UX 디자인 디테일 강화

다크 테마 고도화: 사이드바 및 입력창 색상을 부드러운 다크 그레이(Dark Gray)로 조정
밝기 개선: 배경 오버레이 투명도를 조절하여 전체적으로 화사한 분위기 연출
헤더 투명화: 상단 Deploy 영역 배경을 투명하게 처리하여 일체감 형성
4. 챗봇 인터페이스 최적화

토글 버튼 기능 유지: 우측 하단 버튼(💬/✖)의 직관적인 열기/닫기 인터랙션 유지
버튼 충돌 해결: 플로팅 버튼 스타일이 사이드바 버튼에 영향을 주지 않도록 CSS 스코프 분리 (!important 오버라이딩 적용)
