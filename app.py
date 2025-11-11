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
# 🧠 CONFIGURACIÓN
# ==============================================================
st.set_page_config(page_title="FastMind", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Tu coach de ayuno inteligente — powered by Ekilibrium Technologies")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# ==============================================================
# 📊 FASES DE AYUNO
# ==============================================================
@st.cache_data
def load_fasting_data():
    return pd.read_csv("fastmind_fasting_phases_en.csv")

data = load_fasting_data()

def get_phase(hours):
    phase = data[(data["fase_inicio_h"] <= hours) & (data["fase_fin_h"] > hours)]
    return data.iloc[-1] if phase.empty else phase.iloc[0]


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
# 📚 KNOWLEDGE BASE (PDF)
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
# 💬 CHAT FASTMIND (RAG + GPT fallback)
# ==============================================================
def ask_fastmind(question, hours):
    phase = get_phase(hours)
    kb_text = ""

    if retriever:
        try:
            docs = retriever.invoke(question)
            if docs:
                kb_text = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            print(f"[WARN] RAG error: {e}")

    context = f"""
You are FastMind, a scientific fasting and wellness coach.

Current fasting phase: {phase['keyword']}
Description: {phase['description']}
What to eat: {phase['what_to_eat']}
Symptoms: {phase['symptoms']}
Recommendations: {phase['recommendations']}
Tip: {phase['tip']}

Reference knowledge base:
{kb_text if kb_text.strip() else "No relevant info found."}
"""

    messages = [
        {"role": "system", "content": "Be a concise, motivational, science-based fasting coach."},
        {"role": "user", "content": f"{context}\nUser question: {question}"},
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating response: {e}"


# ==============================================================
# 🕒 TIMER STATE
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
# 🧵 HILO BACKGROUND DEL TIMER
# ==============================================================
def run_timer():
    while st.session_state.running:
        st.session_state.elapsed_hours = (time.time() - st.session_state.start_time) / 3600
        time.sleep(1)


# ==============================================================
# 🎛 CONTROLES DE TIMER
# ==============================================================
col1, col2 = st.columns(2)
if col1.button("▶️ Start"):
    st.session_state.start_time = time.time()
    st.session_state.running = True
    thread = threading.Thread(target=run_timer, daemon=True)
    thread.start()

if col2.button("⏹ Stop"):
    st.session_state.running = False


# ==============================================================
# 🔄 REFRESCO AUTOMÁTICO CADA 3 SEGUNDOS
# ==============================================================
if st.session_state.running:
    time.sleep(3)
    st.rerun()


# ==============================================================
# 🎯 MOSTRAR DIAL
# ==============================================================
if st.session_state.start_time:
    fig, phase = draw_dial(st.session_state.elapsed_hours)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h3 style='text-align:center;'>🌙 {phase['keyword']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#4A4A4A;'>{phase['tip']}</p>", unsafe_allow_html=True)
else:
    st.info("Presiona ▶️ **Start** para comenzar tu ayuno.")


# ==============================================================
# 💬 CHAT
# ==============================================================
st.divider()
st.subheader("💬 Ask FastMind")

question = st.text_input("Ask about fasting, hydration, or mindset:")
if st.button("Ask"):
    hours = st.session_state.elapsed_hours
    with st.spinner("Thinking..."):
        answer = ask_fastmind(question, hours)
    st.session_state.chat_history.append((question, answer))

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
