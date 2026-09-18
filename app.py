
import streamlit as st
import yaml
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TechNova Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TechNova Knowledge Assistant")

st.write(
    "Ask questions about TechNova Solutions policies and documents."
)


# =========================================================
# LOAD CONFIGURATION
# =========================================================

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)


# =========================================================
# CHECK GEMINI API KEY
# =========================================================

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()


# =========================================================
# SESSION MANAGEMENT
# =========================================================

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "active_session" not in st.session_state:
    st.session_state.active_session = None


# =========================================================
# CREATE FIRST CHAT
# =========================================================

if not st.session_state.chat_sessions:

    new_id = str(uuid.uuid4())

    st.session_state.chat_sessions[new_id] = {
        "name": "Chat 1",
        "chat_history": []
    }

    st.session_state.active_session = new_id


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("💬 Chat Settings")


# =========================================================
# NEW CHAT
# =========================================================

if st.sidebar.button("🆕 New Chat"):

    new_id = str(uuid.uuid4())

    chat_number = len(st.session_state.chat_sessions) + 1

    st.session_state.chat_sessions[new_id] = {
        "name": f"Chat {chat_number}",
        "chat_history": []
    }

    st.session_state.active_session = new_id

    st.rerun()


# =========================================================
# CHAT SESSION SELECTOR
# =========================================================

session_names = {
    sid: info["name"]
    for sid, info in st.session_state.chat_sessions.items()
}


selected_name = st.sidebar.selectbox(
    "Active Chat Session",
    options=list(session_names.values())
)


# =========================================================
# FIND SELECTED SESSION ID
# =========================================================

for sid, info in st.session_state.chat_sessions.items():

    if info["name"] == selected_name:

        st.session_state.active_session = sid

        break


# =========================================================
# CURRENT SESSION
# =========================================================

current_session = st.session_state.chat_sessions[
    st.session_state.active_session
]


# =========================================================
# SHOW SESSION ID
# =========================================================

st.sidebar.text(
    f"SESSION ID: {st.session_state.active_session}"
)


# =========================================================
# MEMORY SETTING
# =========================================================

memory_enabled = st.sidebar.toggle(
    "🧠 Enable Memory",
    value=True,
    key=f"memory_{st.session_state.active_session}"
)


# =========================================================
# RAG PIPELINE
# =========================================================

@st.cache_resource
def initialize_rag(session_id):
    from src.ingestion.Document_loader import load_documents
    from src.ingestion.Chunker import chunk_document
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.reranker import CrossEncoderReranker
    from src.generation.llm_generator import GeminiGenerator
    from src.pipeline.rag_pipeline import RAGPipeline

    # -----------------------------------------------------
    # 1. Load documents
    # -----------------------------------------------------

    handbook_docs = load_documents(
        "./data/documents/Employee_Handbook.docx"
    )

    leave_docs = load_documents(
        "./data/documents/Employee_Leave_Policy.pdf"
    )

    benefits_docs = load_documents(
        "./data/documents/Employee_benefits.txt"
    )

    # -----------------------------------------------------
    # 2. Combine documents
    # -----------------------------------------------------

    docs = (
        handbook_docs
        + leave_docs
        + benefits_docs
    )

    # -----------------------------------------------------
    # 3. Create chunks
    # -----------------------------------------------------

    chunks = chunk_document(docs)

    print("Total documents:", len(docs))
    print("Total chunks:", len(chunks))

    # -----------------------------------------------------
    # 4. Hybrid retrieval
    # -----------------------------------------------------

    hybrid = HybridRetriever(
        chunks,
        config_path="./config.yaml"
    )

    # -----------------------------------------------------
    # 5. Cross encoder reranker
    # -----------------------------------------------------

    reranker = CrossEncoderReranker(
        config_path="./config.yaml"
    )

    # -----------------------------------------------------
    # 6. Gemini generator
    # -----------------------------------------------------

    generator = GeminiGenerator()

    # -----------------------------------------------------
    # 7. RAG pipeline
    # -----------------------------------------------------

    rag = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        generator=generator,
        session_id=session_id
    )

    return rag


# =========================================================
# DISPLAY CURRENT CHAT HISTORY
# =========================================================

for message in current_session["chat_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about TechNova..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------

    current_session["chat_history"].append({
        "role": "user",
        "content": question
    })

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # Initialize RAG for current session
    # -----------------------------------------------------

    with st.spinner(
        "🤖 Initializing knowledge assistant..."
    ):

        rag = initialize_rag(
            st.session_state.active_session
        )


    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching documents..."
        ):

            try:

                result = rag.query(question)

                answer = result["answer"]

            except Exception as e:

                answer = f"Error: {str(e)}"


        st.markdown(answer)


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    current_session["chat_history"].append({
        "role": "assistant",
        "content": answer
    })
