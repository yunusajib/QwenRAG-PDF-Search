"""
Qwen2.5 PDF RAG System with ChromaDB, LangChain and Gradio
This script implements a Retrieval-Augmented Generation system for PDF documents
using Qwen2.5 models, ChromaDB for vector storage, and LangChain for the RAG pipeline.
The user interface is built with Gradio.
"""

import os
import time
import argparse
import gradio as gr
from typing import List, Dict, Any, Tuple

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

# LMDeploy for Qwen2.5 models
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

class PDFRagSystem:
    """PDF RAG System using Qwen2.5, ChromaDB, and LangChain"""
    
    def __init__(self, model_name: str, persist_directory: str = "db"):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the Qwen model to use
            persist_directory: Directory to store the vector database
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.pipe = None
        self.vectorstore = None
        self.embeddings = None
        self.top_sources = []  # Store top sources for each query
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load LLM
        self._load_llm()
    
    def change_model(self, model_name: str) -> str:
        """
        Change the LLM model
        
        Args:
            model_name: New model name to use
            
        Returns:
            Status message
        """
        if self.model_name == model_name:
            return f"Already using model: {model_name}"
        
        # Update model name
        self.model_name = model_name
        
        # Reload LLM
        try:
            self._load_llm()
            return f"Successfully switched to model: {model_name}"
        except Exception as e:
            return f"Error switching model: {str(e)}"
        
    def _load_llm(self):
        """Load the Qwen2.5 model with optimized settings"""
        print(f"\nLoading {self.model_name} model...")
        start_time = time.time()
        
        # Configure engine for memory optimization
        engine_config = TurbomindEngineConfig(
            cache_max_entry_count=0.5,  # Use 50% of free GPU memory for KV cache
            session_len=4096            # Reduce context length if memory is limited
        )
        
        # Create the pipeline
        self.pipe = pipeline(self.model_name, backend_config=engine_config)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    def process_pdf(self, pdf_file: str) -> List[Document]:
        """
        Process a PDF file into documents for the vectorstore
        
        Args:
            pdf_file: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        # Load PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, pdf_files: List[str]) -> None:
        """
        Create or update the vector store with documents from PDF files
        
        Args:
            pdf_files: List of paths to PDF files
        """
        all_chunks = []
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"Warning: File {pdf_file} does not exist. Skipping.")
                continue
                
            print(f"Processing {pdf_file}...")
            chunks = self.process_pdf(pdf_file)
            print(f"Created {len(chunks)} chunks from {pdf_file}")
            all_chunks.extend(chunks)
        
        # Create or update vectorstore
        if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
            print("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Adding new documents to existing vectorstore...")
            self.vectorstore.add_documents(all_chunks)
        else:
            print("Creating new vectorstore...")
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        # Persist to disk
        self.vectorstore.persist()
        print(f"Vectorstore created with {len(all_chunks)} chunks and persisted to {self.persist_directory}")
    
    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            k: Number of top documents to retrieve
            
        Returns:
            Tuple of (concatenated context string, list of source documents)
        """
        if not self.vectorstore:
            return "", []
        
        # Search for relevant documents
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format context
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            # Format document content with score
            context_part = f"Document {i+1} [Relevance: {score:.2f}]:\n{doc.page_content}\n"
            context_parts.append(context_part)
            
            # Convert any complex objects to simple types for serialization
            try:
                # Make a clean copy of metadata with only string keys and simple values
                clean_metadata = {}
                for key, value in doc.metadata.items():
                    # Convert key to string
                    str_key = str(key)
                    # Convert value to a simple type
                    if isinstance(value, (str, int, float, bool, type(None))):
                        clean_metadata[str_key] = value
                    else:
                        clean_metadata[str_key] = str(value)
                
                # Prepare source info with clean metadata
                source_info = {
                    "content": str(doc.page_content),
                    "metadata": clean_metadata,
                    "score": float(score),
                    "source_id": i+1
                }
            except Exception as e:
                # Fallback if there's an error creating the source info
                source_info = {
                    "content": str(doc.page_content)[:1000],  # Limit length if it's problematic
                    "metadata": {"error": f"Error processing metadata: {str(e)}"},
                    "score": float(score),
                    "source_id": i+1
                }
            
            sources.append(source_info)
        
        # Store sources for display in UI
        self.top_sources = sources
        
        # Combine all context
        context = "\n".join(context_parts)
        return context, sources
    
    def generate_response(self, query: str, system_prompt: str = "You are a helpful assistant that answers questions based on the provided documents.") -> str:
        """
        Generate a response using RAG
        
        Args:
            query: User query
            system_prompt: System prompt to set assistant behavior
            
        Returns:
            Model response
        """
        # Retrieve relevant context
        context, _ = self.retrieve_context(query)
        
        if not context:
            return "No relevant documents found in the database. Please upload some PDF files first."
        
        # Create RAG prompt
        rag_prompt = f"""Please answer the following question based only on the provided context. If the context doesn't contain relevant information, say that you don't know.

Context:
{context}

Question: {query}"""
        
        # Configure generation parameters
        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=40
        )
        
        # Format in chat-style
        chat_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt}
        ]
        
        # Generate response
        print(f"Running inference for query: {query}")
        start_time = time.time()
        response = self.pipe(chat_prompt, gen_config=gen_config)
        inference_time = time.time() - start_time
        
        # Extract text from response
        if hasattr(response, 'text'):
            result = response.text
        else:
            result = str(response)
            
        print(f"Inference completed in {inference_time:.2f} seconds")
        return result
    
    def get_top_sources(self) -> List[Dict]:
        """Get the top sources used for the last query"""
        return self.top_sources


# Gradio UI Implementation
class RagUI:
    """Gradio UI for the PDF RAG System"""
    
    def __init__(self, rag_system: PDFRagSystem):
        """
        Initialize the UI
        
        Args:
            rag_system: The RAG system to use
        """
        self.rag_system = rag_system
        self.interface = None
        
        # Define model mapping
        self.models = {
            "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct-1M",
            "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
            "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct"
        }
        
        # Get the current model's display name
        self.current_model = next(
            (k for k, v in self.models.items() if v == self.rag_system.model_name),
            "Qwen2.5-3B"  # Default fallback
        )
    
    def _upload_files(self, files: List[str]) -> str:
        """
        Handle file upload
        
        Args:
            files: List of file paths
            
        Returns:
            Status message
        """
        if not files:
            return "No files selected."
        
        try:
            self.rag_system.create_vectorstore([f.name for f in files])
            return f"Successfully processed {len(files)} PDFs."
        except Exception as e:
            return f"Error processing files: {str(e)}"
    
    def _switch_model(self, model_name: str) -> str:
        """
        Switch the model
        
        Args:
            model_name: Name of model to switch to (display name)
            
        Returns:
            Status message
        """
        if model_name not in self.models:
            return f"Unknown model: {model_name}"
        
        # Get the full model name
        full_model_name = self.models[model_name]
        
        # Update the current model
        self.current_model = model_name
        
        # Switch the model in the RAG system
        return self.rag_system.change_model(full_model_name)
    
    def _query(self, query: str, system_prompt: str) -> Tuple[str, List[Dict]]:
        """
        Process a query
        
        Args:
            query: User question
            system_prompt: System prompt to set assistant behavior
            
        Returns:
            Tuple of (response text, sources)
        """
        if not query.strip():
            return "Please enter a question.", []
        
        response = self.rag_system.generate_response(query, system_prompt)
        sources = self.rag_system.get_top_sources()
        
        return response, sources
    
    def _format_source_display(self, sources: List[Dict]) -> str:
        """
        Format sources for display
        
        Args:
            sources: List of source documents
            
        Returns:
            Formatted HTML for display
        """
        if not sources:
            return "<div class='source-container'>No sources available.</div>"
        
        html = "<div class='source-container'>"
        
        # Make sure we're working with actual dictionaries
        for i, source in enumerate(sources):
            try:
                # Handle case where source might not be properly formed
                if not isinstance(source, dict):
                    continue
                    
                # Extract metadata safely
                metadata = source.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    
                page_num = metadata.get("page", "Unknown")
                source_file = metadata.get("source", "Unknown")
                content = source.get("content", "No content available")
                score = source.get("score", 0.0)
                source_id = source.get("source_id", i+1)
                
                # Determine relevance class based on score
                if score >= 0.8:
                    relevance_class = "relevance-high"
                elif score >= 0.6:
                    relevance_class = "relevance-medium"
                else:
                    relevance_class = "relevance-low"
                
                # Format as a card with our CSS classes
                html += f"""
                <div class="source-card">
                    <div class="source-header">
                        Source {source_id} (<span class="{relevance_class}">Relevance: {score:.2f}</span>)
                    </div>
                    <div class="source-meta">
                        <strong>File:</strong> {os.path.basename(str(source_file))}
                    </div>
                    <div class="source-meta">
                        <strong>Page:</strong> {page_num}
                    </div>
                    <div class="source-content">
                        {content}
                    </div>
                </div>
                """
            except Exception as e:
                # Handle any formatting errors
                html += f'<div class="source-card">Error displaying source {i+1}: {str(e)}</div>'
        
        html += "</div>"
        return html
    
    def build_interface(self):
        """Build the Gradio interface"""
        with gr.Blocks(title="Qwen2.5 PDF RAG System") as interface:
            gr.Markdown("# Qwen2.5 PDF RAG System")
            gr.Markdown("Upload PDF files, then ask questions about their content.")
            
            # Model selection section at the top
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model Selection")
                    model_dropdown = gr.Dropdown(
                        choices=list(self.models.keys()),
                        value=self.current_model,
                        label="Select Qwen2.5 Model",
                        info="Larger models are more accurate but slower"
                    )
                    model_status = gr.Textbox(
                        label="Model Status", 
                        value=f"Currently using: {self.current_model}",
                        interactive=False
                    )
                    model_switch_btn = gr.Button("Switch Model", variant="secondary")
            
            with gr.Tab("Upload & Query"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # File upload section
                        gr.Markdown("### Upload PDFs")
                        file_input = gr.File(
                            file_count="multiple",
                            label="Upload PDF Files"
                        )
                        upload_button = gr.Button("Process PDFs", variant="primary")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        
                        # System prompt
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful assistant that answers questions based only on the provided documents. You must cite your sources.",
                            lines=2
                        )
                    
                    with gr.Column(scale=2):
                        # Query section
                        gr.Markdown("### Ask Questions")
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about the uploaded PDFs...",
                            lines=2
                        )
                        query_button = gr.Button("Ask", variant="primary")
                        answer_output = gr.Textbox(
                            label="Answer",
                            interactive=False,
                            lines=10
                        )
            
            # Source Documents Tab
            with gr.Tab("Reference Sources"):
                gr.Markdown("### Sources Used for Answer")
                gr.Markdown("This tab shows the top document chunks that were used to generate the answer.")
                
                # Add some styling to make the display more user-friendly
                gr.HTML("""
                <style>
                .source-container {
                    max-height: 800px;
                    overflow-y: auto;
                    padding: 10px;
                }
                .source-card {
                    margin-bottom: 20px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .source-header {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .source-meta {
                    color: #666;
                    margin-bottom: 8px;
                }
                .source-content {
                    background-color: #f9f9f9;
                    padding: 12px;
                    border-radius: 4px;
                    border-left: 3px solid #2c7be5;
                    font-family: monospace;
                    white-space: pre-wrap;
                    overflow-x: auto;
                }
                .relevance-high {
                    color: #1e7e34;
                }
                .relevance-medium {
                    color: #1f75cb;
                }
                .relevance-low {
                    color: #6c757d;
                }
                </style>
                """)
                
                sources_display = gr.HTML(label="Sources")
            
            # System Info Tab
            with gr.Tab("System Info"):
                gr.Markdown("### System Information")
                gr.Markdown("""
                This PDF RAG (Retrieval-Augmented Generation) system uses:
                
                - **Qwen2.5 Models** for text generation
                - **ChromaDB** for vector storage and similarity search
                - **LangChain** for the RAG pipeline
                
                #### Available Models:
                
                1. **Qwen2.5-1.5B** - Fastest, smallest model for simple queries (1.5 billion parameters)
                2. **Qwen2.5-3B** - Good balance of speed and quality (3 billion parameters)
                3. **Qwen2.5-7B** - Most accurate model for complex questions (7 billion parameters)
                
                #### Memory Usage:
                
                - The 1.5B model requires approximately 3GB of VRAM
                - The 3B model requires approximately 6GB of VRAM
                - The 7B model requires approximately 14GB of VRAM
                
                Model switching happens in real-time and takes a few seconds.
                """)
            
            # Set up events
            upload_button.click(
                fn=self._upload_files,
                inputs=[file_input],
                outputs=[upload_status]
            )
            
            # Define a wrapper function that returns formatted HTML directly
            def query_and_format(query, system_prompt):
                response, sources = self._query(query, system_prompt)
                sources_html = self._format_source_display(sources)
                return response, sources_html
            
            # Use the wrapper function for query events
            query_button.click(
                fn=query_and_format,
                inputs=[query_input, system_prompt],
                outputs=[answer_output, sources_display]
            )
            
            # Also trigger query on pressing Enter in the query input
            query_input.submit(
                fn=query_and_format,
                inputs=[query_input, system_prompt],
                outputs=[answer_output, sources_display]
            )
            
            # Model switching event
            model_switch_btn.click(
                fn=self._switch_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
        
        self.interface = interface
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if not self.interface:
            self.build_interface()
        
        self.interface.launch(**kwargs)


def main():
    """Main function to run the application"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Qwen2.5 PDF RAG System")
    
    # Model selection argument
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["7b", "3b", "1.5b"],
        default="3b",
        help="Model size to use: 7b, 3b, or 1.5b"
    )
    
    # Database directory
    parser.add_argument(
        "--db_dir", 
        type=str, 
        default="chroma_db",
        help="Directory to store the vector database"
    )
    
    # Gradio server settings
    parser.add_argument(
        "--share", 
        action="store_true", default=True,
        help="Create a shareable link"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port to run the Gradio server on"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Define model mapping
    models = {
        "7b": "Qwen/Qwen2.5-7B-Instruct-1M",
        "3b": "Qwen/Qwen2.5-3B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
    }
    
    model_name = models[args.model]
    
    print(f"Starting PDF RAG system with model: {model_name}")
    print(f"Vector database directory: {args.db_dir}")
    
    # Create the RAG system
    rag_system = PDFRagSystem(model_name, args.db_dir)
    
    # Create and launch the UI
    ui = RagUI(rag_system)
    ui.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
