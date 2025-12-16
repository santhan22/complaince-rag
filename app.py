# app.py
import streamlit as st
import tempfile
import os

from rag_query import ask_with_history, SESSION_STORE

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, AIMessage


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Compliance RAG (DPDP Act)",
    layout="wide"
)

st.title("Compliance RAG — Upload documents & chat")


# -------------------------
# Session handling
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_1"

session_id = st.session_state.session_id


# -------------------------
# Sidebar — Upload PDFs
# -------------------------
st.sidebar.header("📄 Upload documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Ingest documents"):
    all_docs = []

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load PDF via LangChain (NO fitz)
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Add filename metadata
        for d in documents:
            d.metadata["source"] = uploaded_file.name

        all_docs.extend(documents)

        os.remove(tmp_path)

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)

    # Build session-only vector store
    with st.spinner("Indexing documents..."):
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        st.session_state.vectordb = vectordb

    st.sidebar.success(f"Ingested {len(uploaded_files)} document(s)")


# -------------------------
# Main chat input
# -------------------------
query = st.text_input("Ask a question about your uploaded documents:")

if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif "vectordb" not in st.session_state:
        st.warning("Please upload and ingest documents first.")
    else:
        retriever = st.session_state.vectordb.as_retriever(
            search_kwargs={"k": 5}
        )
        with st.spinner("Thinking..."):
            answer = ask_with_history(
                query,
                retriever=retriever,
                session_id=session_id
            )

        st.markdown("### Answer")
        st.write(answer)


# -------------------------
# Display chat history
# -------------------------
st.markdown("---")
st.markdown("### 🧠 Conversation Memory")

history = SESSION_STORE.get(session_id)

if history and history.messages:
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.write(f"**User:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.write(f"**Assistant:** {msg.content}")
else:
    st.write("_No messages yet_")


# -------------------------
# Clear session
# -------------------------
if st.sidebar.button("Clear session"):
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
    if "vectordb" in st.session_state:
        del st.session_state.vectordb
    st.success("Session cleared!")

