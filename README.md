📘 QwenRAG-PDF-Search

Ask questions from your PDFs — with real citations and instant context retrieval

QwenRAG-PDF-Search enables users to upload one or multiple PDF documents and interact with them using natural language.
It utilises Retrieval-Augmented Generation (RAG) to return accurate, context-aware answers with source citations, combining semantic search from ChromaDB with Qwen2.5 LLMs.

🎥 Demo Video:
 https://youtu.be/v-ciwaRasvU<img width="468" height="50" alt="image" src="https://github.com/user-                    attachments/assets/48a977e5-c16f-4634-8a91-05fc555e66cd" />

🔹 Problem

Professionals waste valuable time searching through long PDF reports, research papers, and manuals to find information. Traditional search (Ctrl + F) doesn’t handle synonyms or complex questions.

🔹 Solution

This project transforms static document reading into an interactive conversation:

          1️PDFs are uploaded and automatically chunked
          2️Text is embedded with SentenceTransformers
          3️Embeddings stored in ChromaDB for semantic retrieval
          4️Retrieved context is passed to Qwen2.5 via LangChain
          5️The LLM answers while showing exact page citations

🚀 Key Features

        ✔ Upload multiple PDFs
        ✔ Semantic search for improved accuracy
        ✔ Context-aware answers with source text
        ✔ Select model size: 1.5B / 3B / 7B
        ✔ Clean and fast Gradio chat interface
        ✔ Fully local workflow (depending on model used)

📦 Tech Stack
        Component	Tool
        LLM	Qwen2.5 (LMDeploy)
        Document QA	LangChain
        Vector Storage	ChromaDB
        Embeddings	SentenceTransformers (all-MiniLM-L6-v2)
        Interface	Gradio
        Language	Python 3.10+
🖥️ Demo UI

        Add a screenshot 

📂 Project Structure

          - QwenRAG-PDF-Search/
          - app.py
          - requirements.txt
          - README.md
          - src/
               ├── rag_system.py     # Core pipeline: embeddings, retrieval, generation
               └── ui.py             # Gradio interface
          -  data/                 # Uploaded PDFs
          - chroma_db/            # Local vector DB storage
          - assets/               # UI assets (screenshots, icons)

🧠 Skills Demonstrated

        Retrieval-Augmented Generation (RAG)
        
        Vector database design and queries
        
        Python backend development
        
        LLM integration & model selection
        
        Building and deploying interactive AI apps

🛠️ Installation & Run Locally

        git clone 
        cd QwenRAG-PDF-Search
        pip install -r requirements.txt
        python app.py

📊 Evaluation & Future Enhancements

        Planned Feature: Why It Helps
        Response confidence scoring: Improve transparency of answers
        PDF summarisation mode: Faster understanding of long docs
        Chat history memory, Better conversation flow
        Multi-language support, Broader accessibility
        Deployment to HuggingFace/Streamlit	Public demo for portfolio visibility
