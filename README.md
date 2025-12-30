# Multimodal Document RAG

A multimodal retrieval-augmented generation (RAG) system for PDF document understanding using vision-language models.

## Overview

This project enables intelligent querying of PDF documents through a combination of visual embeddings and language models. It converts PDF pages to images, generates embeddings using ColPali, and answers questions using DeepSeek Janus.

## Features

- PDF to image conversion with visual embedding generation
- Vector similarity search for relevant context retrieval
- Multimodal question answering using DeepSeek Janus-Pro
- Interactive web interface built with Gradio
- In-memory vector store for fast retrieval

## Architecture

- **Embedding Model**: ColPali v1.2 for visual document embeddings
- **Language Model**: DeepSeek Janus-Pro-1B for multimodal understanding
- **Vector Store**: In-memory cosine similarity search
- **Interface**: Gradio web UI

## Installation

```bash
pip install torch gradio pdf2image colpali-engine transformers pillow numpy tqdm
```

Additional system requirements:
- Install poppler for PDF processing (required by pdf2image)

## Usage

```bash
python app.py
```

Access the interface at `http://localhost:7860`

### Workflow

1. Upload a PDF document
2. Wait for processing (conversion + embedding generation)
3. Ask questions about the document content

## Project Structure

```
.
├── app.py           # Gradio interface and application logic
├── rag_code.py      # Core RAG components (embeddings, retrieval, generation)
└── cache/           # Temporary storage for processed images
```

## Technical Details

**EmbedData**: Generates visual embeddings from PDF page images using ColPali

**Retriever**: Performs vector similarity search to find relevant pages

**RAG**: Combines retrieval with DeepSeek Janus for context-aware answers

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM

## License

See LICENSE file for details.
