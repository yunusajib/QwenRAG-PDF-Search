# app.py
import argparse
from src.rag_system import PDFRagSystem
from src.ui import RagUI
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5 PDF RAG System")
    parser.add_argument("--model", type=str,
                        choices=["7b", "3b", "1.5b"], default="3b")
    parser.add_argument("--db_dir", type=str, default="chroma_db")
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    models = {
        "7b": "Qwen/Qwen2.5-7B-Instruct-1M",
        "3b": "Qwen/Qwen2.5-3B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
    }

    model_name = models[args.model]
    print(f"🚀 Starting PDF RAG system with model: {model_name}")
    rag_system = PDFRagSystem(model_name, args.db_dir)
    ui = RagUI(rag_system)
    ui.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
