import streamlit as st
import tempfile
import os

from ingest_documents import ingest  # from your original script
from rag_chat import load_vectordb, generate_answer  # from your original script

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("📄 RAG Chatbot with hkunlp/instructor-base & OpenRouter")

# Step 1: File uploader
uploaded_file = st.file_uploader(
    "Upload your document (.txt, .pdf, .html)",
    type=["txt", "pdf", "html"]
)

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Step 2: Process uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    with st.spinner("⚙️ Embedding document... Please wait."):
        ingest(temp_path)

    st.session_state.db_ready = True
    st.session_state.vectordb = load_vectordb()
    st.success("✅ Document embedded and stored in ChromaDB.")

# Step 3: Question answering
if st.session_state.db_ready and st.session_state.vectordb:
    st.subheader("Ask a question about the uploaded document:")
    question = st.text_input("Your question:")

    if st.button("Ask") and question.strip():
        with st.spinner("💭 Thinking..."):
            docs = st.session_state.vectordb.similarity_search(question, k=4)
            answer = generate_answer(docs, question)
        st.markdown(f"**Answer:** {answer}")
