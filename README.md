# Legal RAG & LLM Experiments

This project explores **applied Artificial Intelligence with Large Language Models (LLMs)**,
focusing on legal documents and **Retrieval-Augmented Generation (RAG)** techniques.

The main goal is to experiment with:
- different LLMs (local and API-based)
- semantic search using embeddings
- legal document processing (PDFs)
- RAG pipelines for question answering over laws

> **Note:** Legal documents and datasets used in this project are in Spanish.

---

## Project Structure

- `src/llm_simple/`  
  Simple experiments querying different LLMs (OpenAI, Mistral, Mythomax, Phi-2, Zephyr, etc).

- `src/rag_experiments/`  
  Scripts for PDF processing, dataset generation, and RAG-style experiments.

- `src/datasets/`  
  Legal documents (PDFs) and generated datasets.

- `data/`  
  Evaluation questions and model comparison results.

- `docs/`  
  Technical notes and documentation collected during learning and experimentation.

---

## RAG Pipeline Overview

The legal RAG workflow implemented in this project follows these steps:

1. Extract text from legal PDFs.
2. Split text into semantically meaningful chunks.
3. Generate embeddings using a multilingual sentence-transformer model.
4. Store embeddings in a FAISS vector index.
5. Perform semantic search to retrieve the most relevant text fragments.
6. Inject retrieved context into a prompt.
7. Query a locally hosted LLM through an OpenAI-compatible API.
8. Return a grounded answer based on the legal source documents.

This approach helps reduce hallucinations by grounding model responses in verified legal text.

---

## External Dependency (RAG / LLM Server)

This project depends on a **separately running RAG/LLM server**, based on the open-source project:

**TextGen_Dependencies_For_LLM**

The server must be launched in a separate terminal and expose an
OpenAI-compatible API endpoint, for example:

