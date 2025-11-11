import os
import time
import threading
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI

# --- Optional RAG Support ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False


# ==============================================================
# 🧠 CONFIGURACIÓN INICIAL
# ==============================================================
st.set_page_config(page_title="FastMind", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Tu coach de ayuno inteligente — powered by Ekilibrium Technologies")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# ==============================================================
# 📊 CARGAR DATOS DE FASES DE AYUNO
# ==============================================================
@st.cache_data
def load_fasting_data():
    return pd.read_csv("fastmind_fasting_phases_en.csv")

data = load_fasting_data()

def get_phase(hours):
    phase = data[(data["fase_inicio_h"] <= hours) & (data["fase_fin_h"] > hours)]
    if phase.empty:
        return data.iloc[-1]
    return phase.iloc[0]


# ==============================================================
# ⏳ DIAL DE PROGRESO
# ==============================================================
def draw_dial(hours, total=120):
    phase = get_phase(hours)
    color = phase["color_hex"]
    pct = min((hours / total) * 100, 100)
    label = f"{int(hours)}h {(hours % 1)*60:.0f}m"

    fig = go.Figure(
        go.Pie(
            values=[pct, 100 - pct],
            hole=0.7,
            marker_colors=[color, "#E0E0E0"],
            textinfo="none",
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        annotations=[
            dict(text=label, x=0.5, y=0.5, font_size=26, font_color=color, showarrow=False),
            dict(text=phase["keyword"], x=0.5, y=0.37, font_size=16, showarrow=False),
        ],
    )
    return fig, phase


# ==============================================================
# 📚 CARGAR BASE DE CONOCIMIENTO (PDF)
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    if HAS_LANGCHAIN and os.path.exists("fasting_guide.pdf"):
        loader = PyPDFLoader("fasting_guide.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db = Chroma.from_documents(chunks, embedding, persist_directory="./fastmind_db")
        return db.as_retriever(search_kwargs={"k": 3})
    return None

retriever = load_knowledge_base()


# ==============================================================
# 💬 FUNCIÓN PRINCIPAL DE CHAT
# ==============================================================
def ask_fastmind(question, hours):
    phase = get_phase(hours)
    kb_text = ""

    # --- Buscar en Knowledge Base ---
    if retriever:
        try:
            docs = retriever.invoke(question)
            if docs:
                kb_text = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            print(f"[WARN] RAG error: {e}")

    # --- Si RAG devuelve algo, usarlo como contexto ---
    if kb_text.strip():
        context = f"""
You are FastMind, a scientific fasting and wellness coach.

Current fasting phase: {phase['keyword']}
Description: {phase['description']}
What to eat: {phase['what_to_eat']}
Common symptoms: {phase['symptoms']}
Recommendations: {phase['recommendations']}
Tip: {phase['tip']}

Reference knowledge base:
{kb_text}
"""
    else:
        # --- Fallback directo a GPT si no hay contexto RAG ---
        context = f"""
You are FastMind, a scientific fasting and wellness coach.

Current fasting phase: {phase['keyword']}
Description: {phase['description']}
What to eat: {phase['what_to_eat']}
Common symptoms: {phase['symptoms']}
Recommendations: {phase['recommendations']}
Tip: {phase['tip']}

No knowledge base data available. Answer based on your expertise.
"""

    messages = [
        {"role": "system", "content": "Be a concise, motivational, science-based fasting coach."},
        {"role": "user", "content": f"{context}\nUser question: {question}"},
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error while generating answer: {e}"


# ==============================================================
# 🕒 TIMER CONTROL
# ==============================================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns(2)
if col1.button("▶️ Start"):
    st.session_state.start_time = time.time()
    st.session_state.running = True
if col2.button("⏹ Stop"):
    st.session_state.running = False


# ==============================================================
# 🔄 REFRESCO SUAVE DEL DIAL
# ==============================================================
def background_refresh(interval=3):
    """Mantiene actualizando el dial sin congelar el chat."""
    while st.session_state.running:
        time.sleep(interval)
        st.session_state.refresh_flag = True

if "refresh_flag" not in st.session_state:
    st.session_state.refresh_flag = False
if "refresh_thread" not in st.session_state:
    st.session_state.refresh_thread = None

if st.session_state.running and not st.session_state.refresh_thread:
    thread = threading.Thread(target=background_refresh, daemon=True)
    thread.start()
    st.session_state.refresh_thread = thread

if st.session_state.refresh_flag:
    st.session_state.refresh_flag = False
    st.experimental_rerun()


# ==============================================================
# 🎛 MOSTRAR DIAL DE AYUNO
# ==============================================================
dial_placeholder = st.empty()
if st.session_state.start_time:
    elapsed_hours = (time.time() - st.session_state.start_time) / 3600
    with dial_placeholder.container():
        fig, phase = draw_dial(elapsed_hours)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h3 style='text-align:center;'>🌙 {phase['keyword']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:18px; color:#4A4A4A;'>{phase['tip']}</p>", unsafe_allow_html=True)
else:
    st.info("Presiona ▶️ **Start** para comenzar tu ayuno.")


# ==============================================================
# 💬 SECCIÓN DE CHAT
# ==============================================================
st.divider()
st.subheader("💬 Ask FastMind")

question = st.text_input("Ask a question about fasting, hydration, or mindset:")
if st.button("Ask"):
    hours = ((time.time() - st.session_state.start_time) / 3600) if st.session_state.start_time else 0
    with st.spinner("Thinking..."):
        answer = ask_fastmind(question, hours)
    st.session_state.chat_history.append((question, answer))

# Mostrar historial del chat
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"💡 *FastMind:* {a}")


# ==============================================================
# ✨ FOOTER
# ==============================================================
st.markdown(
    """
---
<div style='text-align:center;'>
    <small>Powered by <b>Ekilibrium Technologies</b> | Built with Streamlit & OpenAI</small>
</div>
""",
    unsafe_allow_html=True,
)
