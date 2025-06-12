# Document Search with RAG (Retrieval Augmented Generation)

A powerful document search system that combines Qwen2.5 language models with RAG technology to provide intelligent, context-aware answers from your PDF documents.

## Quick Start: Setting up the Repository

1. Open terminal in your JarvisLabs workspace/ Open Google Colab as require GPU to run the experiment:
   ```bash
   File > New Launcher > Terminal
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yunusajib/QwenRAG-PDF-Search.git
   ```

3. Navigate to project directory:
   ```bash
   cd document_search_with_rag
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open the Gradio interface (usually http://127.0.0.1:7860)

## Using the Application

### Step 1: Upload Documents
1. Open the main interface
2. Leave default settings for initial use
3. Upload a PDF file (sample provided in repo)
4. Click "Process PDFs"
5. Wait for success message: "Successfully processed 1 PDFs."

### Step 2: Ask Questions
1. Type your question in the query box
2. Click "Ask" or press Enter
3. View the answer generated from your documents
4. Check "Reference Sources" tab to see exactly where the answer came from

### Step 3: Explore Different Models
- Available models:
  - Qwen2.5-7B (Most intelligent, slower)
  - Qwen2.5-3B (Balanced)
  - Qwen2.5-1.5B (Fastest)
- Switch models using the dropdown menu
- Larger models provide better answers but take more time

## Features

- **PDF Processing**: Upload and process multiple PDF documents
- **Intelligent Search**: Uses RAG technology to find relevant information
- **Source Attribution**: See exactly which parts of documents were used
- **Multiple Models**: Choose between different model sizes based on your needs
- **Real-time Processing**: Get answers as you ask questions

## System Requirements

- Python 3.8 or higher
- GPU recommended for better performance
- Minimum 8GB RAM (16GB recommended)
- Storage space for models and document index

   ```

## Importance of RAG Technology

Retrieval Augmented Generation (RAG) is a cutting-edge technology that:
- Enhances accuracy of AI responses by grounding them in your documents
- Reduces hallucinations common in regular language models
- Provides transparency by showing source documents
- Enables organizations to leverage their internal documents effectively

This technology is widely used in:
- Corporate knowledge management
- Document processing systems
- Customer support automation
- Research and analysis tools

## Troubleshooting

Common issues and solutions:
1. "Model loading slow": Try smaller models first
2. "Out of memory": Reduce document size or use smaller model
3. "Processing failed": Check PDF format and file size
4. "No answer generated": Verify document processing was successful

## Next Steps

After mastering basic usage:
1. Try with different types of documents
2. Experiment with different models
3. Adjust system prompt for different use cases
4. Integrate with your own document collection

Congratulations! You've now learned about and used RAG technology, a powerful tool that's transforming how organizations handle document search and knowledge management across Europe and the West.