import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

# --- Optional Knowledge Base (LangChain) ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
st.set_page_config(page_title="FastMind – AI Fasting Tracker", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Your AI-powered fasting companion — created by Ekilibrium Technologies")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Missing OpenAI API key. Set OPENAI_API_KEY in your environment variables.")
    st.stop()
client = OpenAI(api_key=api_key)

# =========================================================
# 📊 FASTING PHASE DATA
# =========================================================
@st.cache_data
def load_phases():
    # Try to load the English CSV
    if os.path.exists("fastmind_fasting_phases_en.csv"):
        df = pd.read_csv("fastmind_fasting_phases_en.csv")

        # Normalize column names (auto-rename)
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect phase range columns
        start_col = next((c for c in df.columns if "start" in c or "inicio" in c), None)
        end_col = next((c for c in df.columns if "end" in c or "fin" in c), None)
        keyword_col = next((c for c in df.columns if "key" in c or "fase" in c), None)

        # Rename dynamically
        if start_col and end_col:
            df.rename(columns={start_col: "start_h", end_col: "end_h"}, inplace=True)
        if keyword_col:
            df.rename(columns={keyword_col: "phase"}, inplace=True)

        # If no color column, assign defaults
        if "color_hex" in df.columns:
            df.rename(columns={"color_hex": "color"}, inplace=True)
        elif "color" not in df.columns:
            df["color"] = ["#F4A261", "#E76F51", "#2A9D8F", "#264653", "#1D3557"]

        # Fallback for text fields
        for col in ["description", "what_to_eat", "symptoms", "recommendations", "tip"]:
            if col not in df.columns:
                df[col] = ""

        return df

    # Default data (if CSV missing)
    return pd.DataFrame([
        {"start_h": 0, "end_h": 4, "phase": "Digest", "description": "Body digests food", "color": "#F4A261"},
        {"start_h": 4, "end_h": 12, "phase": "Fat Burn", "description": "Fat burning starts", "color": "#E76F51"},
        {"start_h": 12, "end_h": 24, "phase": "Deep Burn", "description": "Ketones increase", "color": "#2A9D8F"},
        {"start_h": 24, "end_h": 48, "phase": "Autophagy", "description": "Cell regeneration begins", "color": "#264653"},
        {"start_h": 48, "end_h": 120, "phase": "Ketosis", "description": "Full ketosis achieved", "color": "#1D3557"},
    ])

phases = load_phases()

def get_phase(hours):
    phase = phases[(phases["start_h"] <= hours) & (phases["end_h"] > hours)]
    return phase.iloc[-1] if not phase.empty else phases.iloc[-1]

# =========================================================
# 📚 KNOWLEDGE BASE (RAG)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_kb():
    if HAS_LANGCHAIN and os.path.exists("fasting_guide.pdf"):
        loader = PyPDFLoader("fasting_guide.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        embedding = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma.from_documents(chunks, embedding, persist_directory="./fastmind_db")
        return db.as_retriever(search_kwargs={"k": 3})
    return None

retriever = load_kb()

# =========================================================
# 💬 CHAT AGENT
# =========================================================
def ask_fastmind(question, hours):
    phase = get_phase(hours)
    kb_text = ""
    if retriever:
        try:
            docs = retriever.invoke(question)
            if docs:
                kb_text = "\n\n".join(d.page_content for d in docs)
        except Exception:
            pass

    context = f"""
You are FastMind, a scientific fasting coach.
Current phase: {phase['phase']}
Description: {phase['description']}
Time fasting: {hours:.1f} hours.

Knowledge base info:
{kb_text if kb_text else "No extra context found."}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be a concise, motivational fasting coach using science-backed insights."},
                {"role": "user", "content": f"{context}\n\nUser question: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Chat error: {e}"

# =========================================================
# 🔁 STATE MANAGEMENT
# =========================================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "elapsed_h" not in st.session_state:
    st.session_state.elapsed_h = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# 🧭 INTERFACE (2 Independent Agents)
# =========================================================
tab_timer, tab_chat = st.tabs(["⏱️ Fasting Timer", "💬 FastMind Chatbot"])

# =========================================================
# TIMER AGENT
# =========================================================
with tab_timer:
    st.header("⏱️ Fasting Progress Tracker")

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start", use_container_width=True):
        st.session_state.start_time = time.time()
        st.session_state.running = True

    if col2.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False

    # Calculate elapsed time
    if st.session_state.running and st.session_state.start_time:
        st.session_state.elapsed_h = (time.time() - st.session_state.start_time) / 3600

    hours = st.session_state.elapsed_h
    phase = get_phase(hours)
    color = phase["color"]

    # Circular progress
    pct = min((hours / 120) * 100, 100)
    fig = go.Figure(go.Pie(values=[pct, 100 - pct], hole=0.75, marker_colors=[color, "#E0E0E0"], textinfo="none"))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        annotations=[
            dict(text=f"<b>{phase['phase']}</b>", x=0.5, y=0.5, font_size=28, font_color=color, showarrow=False)
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Digital timer
    elapsed_sec = int(hours * 3600)
    h, m, s = elapsed_sec // 3600, (elapsed_sec % 3600) // 60, elapsed_sec % 60
    st.markdown(
        f"<h1 style='text-align:center; color:{color}; font-weight:bold;'>{h:02d}:{m:02d}:{s:02d}</h1>",
        unsafe_allow_html=True
    )
    st.caption(f"Phase: {phase['phase']} — {phase['description']}")
    st.query_params["refresh"] = str(int(time.time()))

# =========================================================
# CHAT AGENT
# =========================================================
with tab_chat:
    st.header("💬 FastMind Chatbot")

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    question = st.chat_input("Ask something about fasting, hydration, or mindset...")
    if question:
        st.session_state.chat_history.append((question, "thinking..."))
        st.experimental_rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "thinking...":
        last_q = st.session_state.chat_history[-1][0]
        with st.spinner("Thinking..."):
            answer = ask_fastmind(last_q, st.session_state.elapsed_h)
        st.session_state.chat_history[-1] = (last_q, answer)
        st.experimental_rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
---
<div style='text-align:center;'>
    <small>Powered by <b>Ekilibrium Technologies</b> | Built with Streamlit & OpenAI</small>
</div>
""",
    unsafe_allow_html=True
)
