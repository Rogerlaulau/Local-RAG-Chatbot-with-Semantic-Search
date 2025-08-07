import os
import sys
import readline  # for arrow key support in CLI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv
load_dotenv()


# Load your OpenRouter API key

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


CHROMA_DIR = "chroma_db"

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


def load_vectordb():
    embedding_model = InstructorEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    return vectordb


def generate_answer(docs, query):
    context = "\n\n".join(doc.page_content for doc in docs)
    if not context:
        return "No relevant context found."

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",  # or other OpenRouter-supported model
        messages=[
            {"role": "system", "content": "Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )
    return response.choices[0].message.content.strip()

def main():
    print("💬 RAG Chatbot (CLI)")
    print("Type your question (or 'exit' to quit)\n")

    vectordb = load_vectordb()

    while True:
        query = input("🔎 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Get top relevant chunks
        docs = vectordb.similarity_search(query, k=4)
        print("\n📚 Retrieved Chunks:")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.page_content[:200]}...\n")

        
        if not docs:
            print("🤷 No relevant context found.")
            continue

        answer = generate_answer(docs, query)
        print(f"🤖 Bot: {answer}\n")

if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Please set the OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    main()
