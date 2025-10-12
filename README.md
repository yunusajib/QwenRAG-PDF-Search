# 🧠 QwenRAG-PDF-Search  
> AI-powered Document Question Answering using Qwen2.5, LangChain, and ChromaDB

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## 🌟 Overview
QwenRAG-PDF-Search allows users to upload PDF documents and ask questions in natural language.  
The system uses **Retrieval-Augmented Generation (RAG)** — combining **semantic search** (ChromaDB) with **Qwen2.5** language models for context-aware answers.

---

## 🚀 Features
✅ Upload multiple PDFs  
✅ Automatically split and embed content into a vector database  
✅ Ask natural-language questions about your documents  
✅ Cite sources for every answer  
✅ Switch between Qwen2.5-1.5B, 3B, or 7B models  
✅ Simple, interactive Gradio interface  

---

## 🧰 Tech Stack
- **LLMs**: Qwen2.5 (via LMDeploy)
- **Frameworks**: LangChain, Gradio
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **Language**: Python 3.10+

---

## 📊 Folder Structure
```bash
QwenRAG-PDF-Search/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── rag_system.py
│   └── ui.py
├── data/
├── chroma_db/
└── assets/
