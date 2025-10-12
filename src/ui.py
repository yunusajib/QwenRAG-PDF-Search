# src/ui.py
import gradio as gr
from typing import List, Dict, Tuple
import os


class RagUI:
    """Gradio UI for the PDF RAG System"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def process_pdfs(self, files):
        file_paths = [f.name for f in files]
        self.rag_system.process_pdfs(file_paths)
        return f"✅ {len(files)} PDF(s) processed successfully."

    def query(self, question):
        answer, sources = self.rag_system.query(question)
        sources_text = "\n".join(
            [f"- {s['source']} (page {s['page']})" for s in sources])
        return f"**Answer:**\n{answer}\n\n**Sources:**\n{sources_text}"

    def launch(self, share=True, server_port=7860):
        with gr.Blocks(title="QwenRAG PDF Search") as demo:
            gr.Markdown(
                "## 🧠 QwenRAG PDF Search\nUpload PDFs and ask questions.")
            with gr.Tab("Upload PDFs"):
                pdf_input = gr.File(file_count="multiple",
                                    label="Upload PDF Files")
                upload_btn = gr.Button("Process PDFs")
                upload_output = gr.Textbox(label="Status")

                upload_btn.click(self.process_pdfs,
                                 inputs=pdf_input, outputs=upload_output)

            with gr.Tab("Ask Questions"):
                question_input = gr.Textbox(label="Enter your question")
                query_btn = gr.Button("Search")
                answer_output = gr.Markdown()

                query_btn.click(self.query, inputs=question_input,
                                outputs=answer_output)

        demo.launch(share=share, server_port=server_port)
