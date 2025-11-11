import os
import time
import threading
import streamlit as st
import pandas as pd
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
# 🧠 CONFIG
# ==============================================================
st.set_page_config(page_title="FastMind", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Tu coach de ayuno y bienestar — powered by Ekilibrium Technologies")

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
# 📚 BASE DE CONOCIMIENTO (PDF)
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    if HAS_LANGCHAIN and os.path.exists("fasting_guide.pdf"):
        try:
            loader = PyPDFLoader("fasting_guide.pdf")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            db = Chroma.from_documents(chunks, embedding, persist_directory="./fastmind_db")
            return db.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar el PDF: {e}")
    return None

retriever = load_knowledge_base()


# ==============================================================
# 💬 CHAT FASTMIND
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
# 🕒 ESTADO GLOBAL
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
# 🧵 HILO DEL TIMER
# ==============================================================
def run_timer():
    """Actualiza el contador en segundo plano."""
    while st.session_state.running:
        st.session_state.elapsed_hours = (time.time() - st.session_state.start_time) / 3600
        time.sleep(1)


# ==============================================================
# 🌗 INTERFAZ CON DOS TABS
# ==============================================================
tab_timer, tab_chat = st.tabs(["⏱️ Fasting Timer", "💬 FastMind Chatbot"])


# ==============================================================
# ⏱️ TIMER TAB
# ==============================================================
with tab_timer:
    st.header("⏱️ Seguimiento del Ayuno")

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start"):
        st.session_state.start_time = time.time()
        st.session_state.running = True
        thread = threading.Thread(target=run_timer, daemon=True)
        thread.start()

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    # 🔁 Redibujar cada 3 s mientras el timer corre
    if st.session_state.running:
        time.sleep(3)
        st.rerun()

    if st.session_state.start_time:
        phase = get_phase(st.session_state.elapsed_hours)
        color = phase["color_hex"]
        pct = min((st.session_state.elapsed_hours / 120) * 100, 100)
        label = f"{int(st.session_state.elapsed_hours)}h {(st.session_state.elapsed_hours % 1)*60:.0f}m"

        fig = go.Figure(
            go.Pie(
                values=[pct, 100 - pct],
                hole=0.7,
                marker_colors=[color, "#E0E0E0"],
                textinfo="none"
            )
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            annotations=[
                dict(text=label, x=0.5, y=0.5, font_size=26,
                     font_color=color, showarrow=False),
                dict(text=phase["keyword"], x=0.5, y=0.37,
                     font_size=16, showarrow=False)
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<h3 style='text-align:center;'>🌙 {phase['keyword']}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center; color:#4A4A4A;'>{phase['tip']}</p>",
            unsafe_allow_html=True
        )

        # Reloj digital
        elapsed_sec = int(st.session_state.elapsed_hours * 3600)
        h, m, s = elapsed_sec // 3600, (elapsed_sec % 3600) // 60, elapsed_sec % 60
        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{h:02d}:{m:02d}:{s:02d}</h2>",
            unsafe_allow_html=True
        )
    else:
        st.info("Presiona ▶️ **Start** para comenzar tu ayuno.")


# ==============================================================
# 💬 CHAT TAB
# ==============================================================
with tab_chat:
    st.header("💬 Asistente de Ayuno FastMind")

    question = st.text_input("Haz una pregunta sobre ayuno, hidratación o bienestar:")
    if st.button("Preguntar"):
        hours = st.session_state.elapsed_hours
        with st.spinner("Pensando..."):
            answer = ask_fastmind(question, hours)
        st.session_state.chat_history.append((question, answer))

    for q, a in st.session_state.chat_history:
        st.markdown(f"**Tú:** {q}")
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
