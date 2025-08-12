### ✅ `README.md`

````markdown
# 💬 Local RAG Chatbot with LLM Inference via OpenRouter

A local-first Retrieval-Augmented Generation (RAG) chatbot that lets users upload `.txt`, `.pdf`, and `.html` files to query their content using natural language. It uses **Chroma** as the vector store, **OpenAI Embedding API** for semantic search, and routes inference to **OpenRouter.ai** to leverage powerful open-source LLMs.

> ⚡ Designed to run locally on a MacBook (M1, 16GB RAM).

---

## 🧠 Architecture Overview

```mermaid
flowchart TD
    A[User Uploads Files] --> B[Text Extraction (.txt/.pdf/.html)]
    B --> C[Embedding via OpenAI API]
    C --> D[Store Vectors in Chroma DB (local)]
    E[User Query] --> F[Embed Query via OpenAI API]
    F --> G[Retrieve Relevant Chunks from Chroma]
    G --> H[Send Context + Query to LLM via OpenRouter.ai]
    H --> I[LLM Response Returned to User]
````

---

## 📦 Features

* Accepts `.txt`, `.pdf`, `.html` file uploads.
* Embeds documents and queries using OpenAI embeddings.
* Retrieves top-k relevant chunks using ChromaDB.
* Sends prompt + context to OpenRouter-compatible models.
* Local, lightweight, and developer-friendly.

---

## 🚀 Getting Started

### 1. **Clone the repo**

```bash
git clone https://github.com/Rogerlaulau/Local-RAG-Chatbot-with-Semantic-Search.git
cd Local-RAG-Chatbot-with-Semantic-Search
```

### 2. **Set up your Python environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. **Environment Variables**

Create a `.env` file in the root directory with the following:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
CHROMA_DB_PATH=./chroma_db
```

### 4. **Run the Service**

```bash
python3 ingest_documents.py ./context/sample.pdf
python3 rag_chat.py
```
OR
```
streamlit run streamlit_app.py
```
---

## 📂 Project Structure

```bash
.
├── ingest_documents.py     # Handles .txt, .pdf, .html parsing and embed documents into ChromaDB
├── rag_chat.py             # Top-K vector search
├── streamlit_app.py        # Web UI for interaction
├── context                 # Input documents
├── requirements.txt        # Required packages
└── .env                    # API keys and configs
```

---

## 🧠 Example Use Case

> Upload a research paper PDF or a set of HTML documentation files. Ask questions like:
>
> * “What is the main conclusion of this paper?”
> * “List all mentioned APIs from the docs.”
> * “Summarize the key risks discussed.”

---

## 🔧 Tech Stack

| Component         | Tool/Library                           |
| ----------------- | -------------------------------------- |
| Vector DB         | [Chroma](https://www.trychroma.com/)   |
| Embeddings        | OpenAI Embeddings API                  |
| LLM Inference     | [OpenRouter.ai](https://openrouter.ai) |
| Backend Framework | Flask or FastAPI                       |
| File Parsing      | PyMuPDF, BeautifulSoup                 |
| Deployment        | Localhost / MacBook M1                 |

---

## 📌 Future Enhancements

* Local LLM fallback (e.g. llama.cpp or Ollama)
* Streamed responses
* Frontend interface (React/Tailwind)
* Support for `.docx` and `.md`

---

## 🛡️ Disclaimer

This project is for educational and prototyping purposes. API keys are required to interact with external services. Always handle sensitive information securely.

---

## 🧑‍💻 Author

**Roger L.** – 
Passionate about AI, LLM integration, and building impactful developer tools.

---

## 📄 License

MIT License. See `LICENSE` file for details.
