import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

# --- Optional LangChain (RAG) support ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    st.warning("LangChain not installed. The chatbot will still work without a knowledge base.")

# ==============================================================
# CONFIGURATION
# ==============================================================
st.set_page_config(page_title="FastMind – AI Fasting Tracker", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Your intelligent fasting coach — powered by Ekilibrium Technologies")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Missing OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)


# ==============================================================
# LOAD FASTING DATA
# ==============================================================
@st.cache_data
def load_fasting_data():
    try:
        return pd.read_csv("fastmind_fasting_phases_en.csv")
    except FileNotFoundError:
        st.error("⚠️ File 'fastmind_fasting_phases_en.csv' not found.")
        return None

data = load_fasting_data()

def get_phase(hours):
    if data is None:
        return pd.Series({
            "keyword": "Error",
            "description": "Data not loaded",
            "what_to_eat": "",
            "symptoms": "",
            "recommendations": "",
            "tip": "",
            "color_hex": "#FF0000"
        })
    phase = data[(data["fase_inicio_h"] <= hours) & (data["fase_fin_h"] > hours)]
    return data.iloc[-1] if phase.empty else phase.iloc[0]


# ==============================================================
# LOAD KNOWLEDGE BASE (PDF)
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    pdf_path = "fasting_guide.pdf"
    if HAS_LANGCHAIN and os.path.exists(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            embedding = OpenAIEmbeddings(openai_api_key=api_key)
            db = Chroma.from_documents(chunks, embedding, persist_directory="./fastmind_db")
            return db.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.warning(f"Could not load knowledge base: {e}")
    else:
        st.info("No 'fasting_guide.pdf' found. The chatbot will work without RAG context.")
    return None

retriever = load_knowledge_base()


# ==============================================================
# CHATBOT
# ==============================================================
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
You are FastMind, a scientific fasting and wellness coach.
Current fasting phase: {phase['keyword']}
Description: {phase['description']}
What to eat: {phase['what_to_eat']}
Common symptoms: {phase['symptoms']}
Recommendations: {phase['recommendations']}
Tip: {phase['tip']}

Reference knowledge base:
{kb_text if kb_text.strip() else "No relevant info found."}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, science-based fasting coach who motivates users."},
                {"role": "user", "content": f"{context}\nUser question: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


# ==============================================================
# STATE
# ==============================================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "elapsed_hours" not in st.session_state:
    st.session_state.elapsed_hours = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================================================
# UI LAYOUT
# ==============================================================
tab_timer, tab_chat = st.tabs(["⏱️ Fasting Timer", "💬 FastMind Chatbot"])


# ==============================================================
# TIMER TAB
# ==============================================================
with tab_timer:
    st.header("⏱️ Fasting Progress")

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start", use_container_width=True):
        st.session_state.start_time = time.time()
        st.session_state.running = True

    if col2.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False

    if st.session_state.running:
        st.session_state.elapsed_hours = (time.time() - st.session_state.start_time) / 3600
        hours = st.session_state.elapsed_hours
        phase = get_phase(hours)
        color = phase["color_hex"]
        pct = min((hours / 120) * 100, 100)

        fig = go.Figure(
            go.Pie(values=[pct, 100 - pct], hole=0.75,
                   marker_colors=[color, "#E0E0E0"], textinfo="none")
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            annotations=[
                dict(text=f"<b>{phase['keyword']}</b>",
                     x=0.5, y=0.5, font_size=28,
                     font_color=color, showarrow=False)
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

        elapsed_sec = int(hours * 3600)
        h, m, s = elapsed_sec // 3600, (elapsed_sec % 3600) // 60, elapsed_sec % 60
        st.markdown(
            f"<h1 style='text-align:center; color:{color}; font-weight:bold;'>{h:02d}:{m:02d}:{s:02d}</h1>",
            unsafe_allow_html=True
        )

        # Lightweight refresh trick (keeps Render stable)
        st.experimental_set_query_params(t=str(int(time.time())))

    else:
        st.info("Press ▶️ **Start** to begin your fast.")
        if st.session_state.elapsed_hours > 0:
            st.metric("Last fast:", f"{st.session_state.elapsed_hours:.2f} h")


# ==============================================================
# CHAT TAB
# ==============================================================
with tab_chat:
    st.header("💬 FastMind Chatbot")

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    question = st.chat_input("Ask something about fasting, hydration, or mindset...")
    if question:
        st.session_state.chat_history.append((question, "typing..."))
        st.experimental_rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "typing...":
        last_question = st.session_state.chat_history[-1][0]
        with st.spinner("Thinking..."):
            hours = st.session_state.elapsed_hours
            answer = ask_fastmind(last_question, hours)
        st.session_state.chat_history[-1] = (last_question, answer)
        st.experimental_rerun()


# ==============================================================
# FOOTER
# ==============================================================
st.markdown(
    """
---
<div style='text-align:center;'>
    <small>Powered by <b>Ekilibrium Technologies</b> | Built with Streamlit & OpenAI</small>
</div>
""",
    unsafe_allow_html=True
)
