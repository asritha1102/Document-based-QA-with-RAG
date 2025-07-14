# 📄 RAG-based Document Question Answering System

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to **ask questions based on uploaded documents** (TXT or PDF). It retrieves the most relevant pieces of content from your document corpus and uses a language model to generate an accurate, context-aware answer.

---

## 🚀 Features

- ✅ Ask questions in natural language
- 📂 Supports `.txt` and `.pdf` documents
- 🔍 Uses FAISS for fast vector similarity search
- 🤖 Powered by open-source embeddings and LLMs (Flan-T5 / MiniLM)
- 🧠 Retrieval-Augmented Generation pipeline (Retriever + Generator)
- 🖥️ Streamlit app with user-friendly interface

---

## 🧠 How It Works (RAG)

1. **Document Loading**: Loads `.txt` or `.pdf` files from the `docs/` folder.
2. **Chunking**: Splits documents into overlapping chunks using LangChain.
3. **Embedding**: Encodes the chunks using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search.
5. **Retrieval**: Retrieves top-k relevant chunks for a user query.
6. **Generation**: Passes retrieved content to an open-source LLM (like `flan-t5-base`) to generate the final answer.

---

## 🛠️ Tech Stack

| Component    | Tool / Model                    |
|--------------|---------------------------------|
| Framework    | [LangChain](https://github.com/langchain-ai/langchain) |
| UI           | Streamlit                       |
| Embeddings   | `all-MiniLM-L6-v2`              |
| Generator    | `flan-t5-base` (via HuggingFace)|
| Vector Store | FAISS                           |
| Language     | Python 3.10+                    |

---

## 📁 Project Structure

rag_qa_project/
├── app.py # Streamlit UI
├── rag_qa.py # Core RAG logic
├── docs/ # Folder to hold documents (.txt/.pdf)
│ └── example.txt
├── .env # (optional for OpenAI API if needed)
└── README.md
