# Legal RAG & LLM Experiments

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![AI](https://img.shields.io/badge/Focus-LLM%20%2B%20RAG-purple)
![Domain](https://img.shields.io/badge/Domain-Legal%20NLP-lightgrey)
![License](https://img.shields.io/github/license/Kairix0/llm_rag_legal_project)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

![Last Commit](https://img.shields.io/github/last-commit/Kairix0/llm_rag_legal_project)
![Repo Size](https://img.shields.io/github/repo-size/Kairix0/llm_rag_legal_project)
![Stars](https://img.shields.io/github/stars/Kairix0/llm_rag_legal_project?style=social)

This project explores **applied Artificial Intelligence with Large Language Models (LLMs)**,
focusing on legal documents and **Retrieval-Augmented Generation (RAG)** techniques.

The main goal is to experiment with:
- different LLMs (local and API-based)
- semantic search using embeddings
- legal document processing (PDFs)
- RAG pipelines for question answering over laws

> **Note:** Legal documents and datasets used in this project are in Spanish.

---

## Project Context

This repository contains **experimental prototypes and technical explorations**
derived from a larger idea: building a Legal AI assistant to support lawyers in
case analysis and trial preparation using LLMs and RAG techniques.

The original project was developed collaboratively but was eventually paused due
to hardware and funding constraints required for large-scale model training.

The code in this repository is **not a production system**, but a collection of
working prototypes that demonstrate:

- PDF ingestion and preprocessing for legal documents
- Embedding generation for Spanish legal text
- Semantic search using FAISS
- Interaction with local and API-based LLMs
- Early RAG-style pipelines for legal question answering

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

