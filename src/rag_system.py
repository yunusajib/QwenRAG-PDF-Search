# src/rag_system.py
import os
import time
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import pipeline
import torch


class PDFRagSystem:
    """PDF RAG System using Qwen2.5, ChromaDB, and LangChain"""

    def __init__(self, model_name: str, db_dir: str = "chroma_db"):
        self.model_name = model_name
        self.db_dir = db_dir
        self.pipeline = None
        self.vectorstore = None
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

    def load_model(self):
        """Load Qwen model via transformers pipeline"""
        print(f"Loading model: {self.model_name} ...")
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✅ Model loaded successfully")

    def process_pdfs(self, pdf_files: List[str]):
        """Extract text, split into chunks, and build ChromaDB index"""
        docs = []
        for pdf_path in pdf_files:
            print(f"📘 Processing: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            chunks, self.embedding_model, persist_directory=self.db_dir)
        self.vectorstore.persist()
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB at {self.db_dir}")

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Retrieve top-k chunks and generate answer"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded. Process PDFs first.")

        docs = self.vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"

        if not self.pipeline:
            self.load_model()

        response = self.pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.8,
            do_sample=True
        )
        answer = response[0]['generated_text'] if isinstance(
            response, list) else response
        sources = [{"source": doc.metadata.get(
            "source", "unknown"), "page": doc.metadata.get("page", "N/A")} for doc in docs]

        return answer, sources
