import streamlit as st
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.Document_loader import load_documents
from src.ingestion.Chunker import chunk_document


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="TechNova Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TechNova Knowledge Assistant")

st.write(
    "Ask questions about TechNova Solutions policies and documents."
)


# -----------------------------
# Load configuration
# -----------------------------

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)


# -----------------------------
# Initialize RAG pipeline
# -----------------------------
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()
@st.cache_resource
def initialize_rag():

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.reranker import CrossEncoderReranker
    from src.generation.llm_generator import GeminiGenerator
    from src.pipeline.rag_pipeline import RAGPipeline

    # 1. Load all documents

    # 1. Load all documents

    handbook_docs = load_documents(
        "./data/documents/Employee_Handbook.docx"
    )

    leave_docs = load_documents(
        "./data/documents/Employee_Leave_Policy.pdf"
    )

    benefits_docs = load_documents(
        "./data/documents/Employee_benefits.txt"
    )

    # 2. Combine documents

    docs = (
        handbook_docs
        + leave_docs
        + benefits_docs
    )

    # 3. Chunk all documents

    chunks = chunk_document(docs)

    print("Total documents:", len(docs))
    print("Total chunks:", len(chunks))

    # 4. Hybrid retrieval

    hybrid = HybridRetriever(
        chunks,
        config_path="./config.yaml"
    )

    # 5. Reranker

    reranker = CrossEncoderReranker(
        config_path="./config.yaml"
    )

    # 6. Gemini generator

    generator = GeminiGenerator()

    # 7. RAG pipeline

    rag = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        generator=generator,
        session_id="streamlit_session"
    )

    return rag


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------

question = st.chat_input(
    "Ask a question about TechNova..."
)


if question:

    # Initialize RAG only when the first question is asked
    with st.spinner("Initializing knowledge assistant..."):

        rag = initialize_rag()

    # -----------------------------
    # Display user question
    # -----------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # -----------------------------
    # Generate answer
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            result = rag.query(question)

        answer = result["answer"]

        st.markdown(answer)


    # -----------------------------
    # Save assistant response
    # -----------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })