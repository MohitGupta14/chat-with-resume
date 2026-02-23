"""
app.py
======
Streamlit Frontend — Chat with Resume
---------------------------------------
A clean, professional UI that:
1. Lets you upload a PDF resume
2. Shows real-time ingestion status
3. Provides a chat interface to ask questions
4. Displays source citations from the resume
"""

import streamlit as st
import requests
import uuid
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_URL = f"http://{os.getenv('FASTAPI_HOST', 'localhost')}:{os.getenv('FASTAPI_PORT', 8000)}"

st.set_page_config(
    page_title="Resume Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — clean dark professional theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root theme */
:root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface2: #252840;
    --accent: #6C63FF;
    --accent2: #FF6584;
    --text: #e8e8f0;
    --muted: #8888aa;
    --border: rgba(108, 99, 255, 0.2);
    --success: #4ade80;
    --error: #f87171;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-header p {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
}

/* Chat messages */
.message-human {
    background: linear-gradient(135deg, #6C63FF22, #6C63FF11);
    border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-left: 3rem;
    position: relative;
}
.message-ai {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-right: 3rem;
}
.message-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Source citations */
.source-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
}
.source-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-ok { background: rgba(74, 222, 128, 0.15); color: var(--success); border: 1px solid rgba(74, 222, 128, 0.3); }
.status-err { background: rgba(248, 113, 113, 0.15); color: var(--error); border: 1px solid rgba(248, 113, 113, 0.3); }

/* Sidebar */
.sidebar-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent), #9c6fff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Input */
.stTextInput input, .stChatInput textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "resume_loaded" not in st.session_state:
    st.session_state.resume_loaded = False

if "namespace" not in st.session_state:
    st.session_state.namespace = "default"

if "resume_name" not in st.session_state:
    st.session_state.resume_name = None


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_resume(file, namespace: str) -> dict:
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    params = {"namespace": namespace}
    response = requests.post(f"{API_URL}/upload", files=files, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "namespace": st.session_state.namespace,
    }
    response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem">📄</div>
        <div style="font-family: 'DM Serif Display', serif; font-size: 1.3rem; 
                    background: linear-gradient(135deg, #6C63FF, #FF6584);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Resume Chat
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # API Status
    api_ok = check_api_health()
    if api_ok:
        st.markdown('<span class="status-badge status-ok">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-err">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Start the API: `python api.py`", icon="⚠️")
    
    st.divider()
    
    # Upload Section
    st.markdown('<div class="sidebar-title">📤 Upload Resume</div>', unsafe_allow_html=True)
    
    namespace = st.text_input("Namespace (optional)", value="default", 
                               help="Use different namespaces to store multiple resumes")
    
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file and api_ok:
        if st.button("⚡ Ingest Resume", use_container_width=True):
            with st.spinner("Chunking & embedding... (may take 20-30s)"):
                try:
                    result = upload_resume(uploaded_file, namespace)
                    st.session_state.resume_loaded = True
                    st.session_state.namespace = namespace
                    st.session_state.resume_name = uploaded_file.name
                    st.session_state.messages = []  # reset chat
                    st.success(f"✅ {result['chunks_count']} chunks ingested!")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
    
    # Resume status
    if st.session_state.resume_loaded:
        st.markdown(f"""
        <div style="background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3);
                    border-radius: 8px; padding: 0.6rem; margin-top: 0.5rem; font-size: 0.82rem;">
            ✅ <strong>{st.session_state.resume_name}</strong><br>
            <span style="color: #8888aa;">Namespace: {st.session_state.namespace}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Suggested Questions
    st.markdown('<div class="sidebar-title">💡 Try These Questions</div>', unsafe_allow_html=True)
    
    suggestions = [
        "Summarize this person's experience",
        "What technical skills do they have?",
        "Where did they study?",
        "What are their top achievements?",
        "Are they a good fit for a backend role?",
    ]
    
    for q in suggestions:
        if st.button(q, key=f"sug_{q}", use_container_width=True):
            st.session_state.pending_question = q
    
    st.divider()
    
    # Session info
    st.markdown(f"""
    <div style="font-size: 0.72rem; color: var(--muted);">
        Session ID: <code>{st.session_state.session_id}</code><br>
        Messages: {len(st.session_state.messages)}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1>Chat with Resume</h1>
    <p>Upload a PDF resume and ask anything — powered by GPT-4 & Pinecone</p>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.resume_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; 
                    border: 2px dashed rgba(108,99,255,0.3); border-radius: 16px;
                    background: rgba(108,99,255,0.04);">
            <div style="font-size: 3rem; margin-bottom: 1rem">📎</div>
            <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem">No resume loaded yet</div>
            <div style="color: #8888aa; font-size: 0.9rem">
                Upload a PDF resume in the sidebar<br>to start chatting with it
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; color: #8888aa; padding: 2rem; font-size: 0.9rem;">
                💬 Ask your first question about the resume below
            </div>
            """, unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            if msg["role"] == "human":
                st.markdown(f"""
                <div class="message-human">
                    <div class="message-label">You</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-ai">
                    <div class="message-label">🤖 AI Assistant</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source citations
                if msg.get("sources"):
                    with st.expander(f"📚 View {len(msg['sources'])} source sections", expanded=False):
                        st.markdown('<div class="source-header">Retrieved from Resume</div>', unsafe_allow_html=True)
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Section {i} — Page {src['page'] + 1}</strong><br>
                                {src['text'][:300]}{'...' if len(src['text']) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)
    
    # Handle suggested question click
    pending = st.session_state.pop("pending_question", None)
    
    # Chat Input
    user_input = st.chat_input("Ask anything about the resume...", key="chat_input")
    
    question = pending or user_input
    
    if question and api_ok:
        # Add human message immediately
        st.session_state.messages.append({"role": "human", "content": question})
        
        with st.spinner("Searching resume & generating answer..."):
            try:
                result = ask_question(question)
                st.session_state.messages.append({
                    "role": "ai",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "ai",
                    "content": f"❌ Error: {str(e)}",
                    "sources": [],
                })
        
        st.rerun()
    
    elif question and not api_ok:
        st.error("⚠️ API is not running. Please start the backend first.")
