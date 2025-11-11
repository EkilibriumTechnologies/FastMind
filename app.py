import os
import time
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
# 🧠 CONFIGURATION
# ==============================================================
# Bind Streamlit to Render’s dynamic port
os.environ["PORT"] = os.environ.get("PORT", "10000")

st.set_page_config(page_title="FastMind", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.markdown(
    "Your personal AI fasting coach and wellness companion — powered by Ekilibrium Technologies."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ==============================================================
# 📊 LOAD FASTING DATA
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
# 📈 DIAL DRAWER
# ==============================================================
def draw_dial(hours, total=120):
    """Display circular fasting progress with smooth animation."""
    phase = get_phase(hours)
    color = phase["color_hex"]
    pct = min((hours / total) * 100, 100)
    label = f"{int(hours)}h {(hours % 1) * 60:.0f}m"

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
            dict(
                text=label,
                x=0.5,
                y=0.5,
                font_size=28,
                font_color=color,
                showarrow=False,
            ),
            dict(
                text=phase["keyword"],
                x=0.5,
                y=0.38,
                font_size=16,
                showarrow=False,
            ),
        ],
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig, phase

# ==============================================================
# 📚 OPTIONAL KNOWLEDGE BASE (PDF)
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
# 💬 ASK FASTMIND
# ==============================================================
def ask_fastmind(question, hours):
    phase = get_phase(hours)
    kb_text = ""

    if retriever:
        docs = retriever.invoke(question)
        kb_text = "\n\n".join(d.page_content for d in docs)

    # fixed triple-quote structure (previous code broke here)
    context = f"""
You are FastMind, a scientific fasting and wellness coach.

Current fasting phase: {phase['keyword']}
Description: {phase['description']}
What to eat: {phase['what_to_eat']}
Common symptoms: {phase['symptoms']}
Recommendations: {phase['recommendations']}
Tip: {phase['tip']}

Reference knowledge base (if available):
{kb_text}
"""

    messages = [
        {
            "role": "system",
            "content": "Be a science-based, motivational fasting coach. Combine clarity, empathy, and data.",
        },
        {"role": "user", "content": f"{context}\nUser question: {question}"},
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error while generating response: {e}"

# ==============================================================
# 🕒 FASTING TIMER LOGIC
# ==============================================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
if col1.button("▶️ Start"):
    st.session_state.start_time = time.time()
    st.session_state.running = True
if col2.button("⏹ Stop"):
    st.session_state.running = False

# ==============================================================
# 🔄 LIVE DISPLAY
# ==============================================================
if st.session_state.start_time:
    elapsed_hours = (time.time() - st.session_state.start_time) / 3600
    fig, phase = draw_dial(elapsed_hours)

    st.markdown(f"<h3 style='text-align:center;'>🌙 Phase: {phase['keyword']}</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; font-size:18px; color:#4A4A4A;'>{phase['tip']}</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.running:
        time.sleep(3)
        st.rerun()
else:
    st.info("Press ▶️ **Start** to begin tracking your fast.")

# ==============================================================
# 💬 CHAT SECTION
# ==============================================================
st.divider()
st.subheader("💬 Ask FastMind")
question = st.text_input("Ask a question about fasting, hydration, or mindset:")
if st.button("Ask"):
    hours = ((time.time() - st.session_state.start_time) / 3600) if st.session_state.start_time else 0
    with st.spinner("Thinking..."):
        answer = ask_fastmind(question, hours)
    st.success(answer)

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
