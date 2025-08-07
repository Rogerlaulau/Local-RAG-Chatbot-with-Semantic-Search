import os
import sys
from langchain.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from InstructorEmbedding import INSTRUCTOR

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings



class InstructorEmbeddings(Embeddings):
    def __init__(self):
        self.model = INSTRUCTOR('hkunlp/instructor-base')
        self.instruction = "Represent this sentence for retrieval:"

    def embed_documents(self, texts):
        pairs = [[self.instruction, text] for text in texts]
        return [embedding.tolist() for embedding in self.model.encode(pairs)]

    def embed_query(self, text):
        embedding = self.model.encode([[self.instruction, text]])
        return embedding[0].tolist()  # flatten numpy array into plain list

# === CONFIG ===
CHROMA_DIR = "chroma_db"

# === LOAD FILE ===
def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PDFPlumberLoader(file_path).load()
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# === EMBEDDING MODEL ===
print("⚙️ Loading InstructorEmbedding model...")
embedder = INSTRUCTOR('hkunlp/instructor-base')

def embed_texts(texts):
    # texts is list of strings
    # InstructorEmbedding expects list of [instruction, text] pairs
    pairs = [["Represent this sentence for retrieval:", t] for t in texts]
    return embedder.encode(pairs)

# === MAIN ===
def ingest(file_path):
    print(f"📥 Loading: {file_path}")
    docs = load_file(file_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")

    # Use wrapped embedding model
    embedding_model = InstructorEmbeddings()

    # Store in Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"✅ Done. Chroma vector store saved to '{CHROMA_DIR}'.")


# === CLI ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_documents.py ./context/<file_path>")
        sys.exit(1)
    ingest(sys.argv[1])
