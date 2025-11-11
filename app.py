import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh # Importante para el timer

# --- Carga de Langchain (RAG) ---
# (Asegúrate de tener el PDF en el mismo directorio)
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    st.error("Faltan algunas librerías de LangChain. Instálalas con `pip install langchain-community langchain-openai chromadb pypdf tiktoken`")

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
st.set_page_config(page_title="FastMind", layout="centered")
st.title("🧠 FastMind – AI Fasting Tracker")
st.caption("Tu coach de ayuno y bienestar — Creado con Streamlit")

# Configura el cliente de OpenAI (lee la API key de los "Secrets" de Streamlit o variables de entorno)
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception:
    st.error("No se encontró la OPENAI_API_KEY. Por favor, configúrala en las variables de entorno de Render.")
    st.stop()


# ==============================================================
# CARGA DE DATOS (CSV y PDF)
# ==============================================================

# Carga las fases del ayuno desde el CSV
@st.cache_data
def load_fasting_data():
    # Usa el nombre de archivo exacto que subiste
    try:
        return pd.read_csv("fastmind_fases_ayuno.csv")
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo `fastmind_fases_ayuno.csv`. Asegúrate de que esté en el repositorio.")
        return None

data = load_fasting_data()

def get_phase(hours):
    if data is None:
        # Retorna un objeto 'dummy' si el CSV no se pudo cargar
        return pd.Series({
            "keyword": "Error", "descripcion": "Data no cargada", 
            "que_comer": "", "sintomas": "", "recomendaciones": "", 
            "tip": "", "color_hex": "#FF0000"
        })
    
    phase = data[(data["fase_inicio_h"] <= hours) & (data["fase_fin_h"] > hours)]
    return data.iloc[-1] if phase.empty else phase.iloc[0]


# Carga la base de conocimiento (PDF) para RAG
@st.cache_resource(show_spinner="Cargando base de conocimiento (PDF)...")
def load_knowledge_base():
    pdf_path = "fasting_guide.pdf" # El PDF que subiste
    if HAS_LANGCHAIN and os.path.exists(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            # Asegúrate de que la API key esté disponible para los embeddings
            if "OPENAI_API_KEY" not in os.environ:
                st.error("OPENAI_API_KEY no encontrada, no se pueden crear embeddings.")
                return None
                
            embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            
            # Advertencia sobre el sistema de archivos de Render
            st.warning("Nota: En el plan gratuito de Render, la base de datos RAG se reconstruirá en cada reinicio, lo que puede ser lento.", icon="⚠️")
            
            # Usar un directorio persistente (si Render lo soporta) o en memoria
            db = Chroma.from_documents(chunks, embedding, persist_directory="./fastmind_db")
            return db.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.error(f"⚠️ No se pudo cargar el PDF para RAG: {e}")
    elif not os.path.exists(pdf_path):
        st.warning(f"No se encontró el archivo `{pdf_path}`. El chatbot funcionará sin RAG.", icon="ℹ️")
    return None

retriever = load_knowledge_base()


# ==============================================================
# LÓGICA DEL CHATBOT
# ==============================================================
def ask_fastmind(question, hours):
    phase = get_phase(hours)
    kb_text = ""
    
    # Intenta obtener contexto RAG desde el PDF si el retriever está disponible
    if retriever:
        try:
            docs = retriever.invoke(question)
            if docs:
                kb_text = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            st.warning(f"Error al consultar la base de datos RAG: {e}")
            pass

    # Contexto del sistema para el LLM
    context = f"""
Eres FastMind, un coach científico de ayuno y bienestar. Responde siempre en español.

---
INFORMACIÓN DE LA FASE ACTUAL DE AYUNO (Horas: {hours:.2f}):
- Fase: {phase['keyword']}
- Descripción: {phase['descripcion']}
- Qué beber/comer: {phase['que_comer']}
- Síntomas comunes: {phase['sintomas']}
- Recomendaciones: {phase['recomendaciones']}
- Tip: {phase['tip']}
---
INFORMACIÓN ADICIONAL DE LA BASE DE CONOCIMIENTO (PDF):
{kb_text if kb_text.strip() else "No se encontró información relevante en el PDF."}
---
"""
    # Prepara el historial para la API de OpenAI
    messages = [
        {"role": "system", "content": "Eres un coach de ayuno motivador, conciso y basado en ciencia. Responde en español."},
        {"role": "user", "content": f"{context}\n\nPregunta del usuario: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Puedes cambiar el modelo si lo deseas
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ Error al generar respuesta de OpenAI: {e}")
        return "Lo siento, tuve un problema al conectar con mi cerebro (OpenAI). Revisa la API key."


# ==============================================================
# MANEJO DE ESTADO (SESSION STATE)
# Esta es la clave para que el chat no se borre.
# ==============================================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "elapsed_hours" not in st.session_state:
    st.session_state.elapsed_hours = 0
if "chat_history" not in st.session_state:
    # El historial de chat persiste entre refrescos
    st.session_state.chat_history = [] 

# ==============================================================
# INTERFAZ (Tabs)
# ==============================================================
tab_timer, tab_chat = st.tabs(["⏱️ Temporizador", "💬 FastMind Chatbot"])

# --- TIMER TAB ---
with tab_timer:
    st.header("⏱️ Seguimiento del Ayuno")

    col1, col2 = st.columns(2)
    if col1.button("▶️ Empezar", use_container_width=True):
        st.session_state.start_time = time.time()
        st.session_state.running = True
        st.rerun() # Refresca inmediatamente al empezar

    if col2.button("⏹ Detener", use_container_width=True):
        st.session_state.running = False
        st.rerun() # Refresca inmediatamente al parar

    # Contenedor para el reloj y el gráfico (se actualizará por el auto-refresh)
    timer_placeholder = st.empty()

    if st.session_state.running and st.session_state.start_time:
        # --- LÓGICA DEL AUTO-REFRESH ---
        # Esto refrescará la app cada segundo *solo* si el timer está corriendo
        # ¡No afectará al session_state!
        st_autorefresh(interval=1000, limit=None, key="timer_refresh")

        # Calcula el tiempo
        elapsed_sec = int(time.time() - st.session_state.start_time)
        st.session_state.elapsed_hours = elapsed_sec / 3600
        hours = st.session_state.elapsed_hours
        phase = get_phase(hours)
        color = phase["color_hex"]
        pct = min((hours / 120) * 100, 100) # Asume un máximo de 120h

        with timer_placeholder.container():
            # Gráfico de Plotly (Dial)
            fig = go.Figure(
                go.Pie(
                    values=[pct, 100 - pct], 
                    hole=0.75,
                    marker_colors=[color, "#4A5568"], 
                    textinfo="none",
                    direction="clockwise",
                    sort=False
                )
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{phase['keyword']}</b>", 
                    x=0.5, y=0.5,
                    font_size=28, 
                    font_color=color, 
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Reloj Digital (Cronómetro)
            h, m, s = elapsed_sec // 3600, (elapsed_sec % 3600) // 60, elapsed_sec % 60
            st.markdown(
                f"<h1 style='text-align:center; color:{color}; font-weight:bold; font-size: 3rem; margin-top: -20px;'>{h:02d}:{m:02d}:{s:02d}</h1>",
                unsafe_allow_html=True
            )
            
            # Detalles de la fase
            with st.expander(f"Detalles de la fase: **{phase['keyword']}**"):
                st.markdown(f"**Descripción:** {phase['descripcion']}")
                st.markdown(f"**Qué beber:** {phase['que_comer']}")
                st.markdown(f"**Síntomas:** {phase['sintomas']}")
                st.markdown(f"**Tip:** {phase['tip']}")

    else:
        # Estado inicial o detenido
        with timer_placeholder.container():
            st.info("Presiona ▶️ **Empezar** para comenzar tu ayuno.")
            if st.session_state.elapsed_hours > 0:
                st.metric("Último ayuno completado:", f"{st.session_state.elapsed_hours:.2f} horas")


# --- CHAT TAB ---
with tab_chat:
    st.header("💬 Asistente de Ayuno FastMind")
    
    # El historial de chat se dibuja desde el session_state
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # Input de chat
    question = st.chat_input("Haz una pregunta sobre tu ayuno...")
    
    if question:
        # Añadir pregunta al historial y mostrarla
        st.session_state.chat_history.append((question, "typing..."))
        st.rerun() # Refresca para mostrar la pregunta inmediatamente

    # Si el último mensaje es "typing...", genera la respuesta
    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "typing...":
        
        # Obtiene la pregunta que acabamos de hacer
        last_question = st.session_state.chat_history[-1][0]
        
        # Genera la respuesta
        with st.spinner("Pensando..."):
            hours = st.session_state.elapsed_hours
            answer = ask_fastmind(last_question, hours)
        
        # Reemplaza "typing..." con la respuesta real
        st.session_state.chat_history[-1] = (last_question, answer)
        st.rerun() # Refresca para mostrar la respuesta

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
    unsafe_allow_html=True,
)
