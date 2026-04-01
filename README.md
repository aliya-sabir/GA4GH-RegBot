# GA4GH RegBot

LLM-powered compliance assistant for checking genomic data consent forms against GA4GH policy and consent-clause documents.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GSoC](https://img.shields.io/badge/GSoC-2026-orange)](https://summerofcode.withgoogle.com/)

## Status

Work in progress. This repository is an active proof of concept and the pipeline, evaluation flow, and documentation may continue to change as development progresses.

## What it does

RegBot takes consent form text, retrieves the most relevant GA4GH clauses using hybrid search, and asks an LLM to produce a structured compliance assessment with clause-backed citations.

The current proof of concept includes:

- PDF ingestion for the bundled GA4GH corpus
- hybrid retrieval with embeddings and BM25
- optional cross-encoder reranking
- JSON compliance output with citations

## Problem it solves

Researchers often have to manually compare consent forms against dense GA4GH policy documents before sharing genomic data. RegBot is meant to make that review faster, more consistent, and easier to trace back to source clauses.

## Included GA4GH sources

The repository already includes the PDFs used for ingestion under `src/docs`:

- GA4GH Framework for Responsible Sharing
- GA4GH Data Privacy and Security Policy
- GA4GH Consent Policy
- GA4GH Clinical Genomic Consent Clauses
- GA4GH Consent Clauses for Genomic Research
- GA4GH Large-Scale Initiatives Consent Clauses
- GA4GH Pediatric Consent to Genetic Research Clauses

Document metadata and source URLs are defined in `src/pdf_sources.json`.

## How it works

1. PDFs are parsed into clause-aware chunks, with table extraction for consent toolkit documents.
2. Chunks are stored in ChromaDB and indexed for semantic search plus BM25 retrieval.
3. Retrieved clauses are optionally reranked with a cross-encoder.
4. The top clauses are sent to a Hugging Face Inference API model for structured compliance analysis.

Example output shape:

```json
{
  "status": "Compliant | Partial | Non-Compliant",
  "missing_elements": ["specific gap in consent form language"],
  "suggested_fix": "1. numbered specific action. 2. numbered specific action.",
  "citations": [
    {
      "citation": "human-readable clause label",
      "source_url": "source link",
      "title": "clause title",
      "excerpt": "retrieved clause text"
    }
  ]
}
```

## Setup

```bash
git clone https://github.com/aliya-sabir/GA4GH-RegBot.git
cd GA4GH-RegBot
pip install -r requirements.txt
```

Create a `.env` file with:

```env
HF_TOKEN=your_huggingface_token
```

Optional environment variables supported by the current code:

- `EMBEDDING_MODEL` to override the default embedding model (`BAAI/bge-small-en-v1.5`)
- `RERANKER_MODEL` to override the default reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `DISABLE_RERANKER=1` to skip reranking
- `TEST_SCENARIO_ID` to choose a bundled evaluation case

## Running the prototype

The most reliable way to run the current pipeline is from Python so you can pass the docs directory explicitly:

```python
from src.run_test import run_test_scenario

run_test_scenario(
    scenario_id="scenario_4",
    docs_path="src/docs",
)
```

Bundled evaluation scenarios live in `src/evaluation/tests.jsonl`.

If you want to work with the main orchestration class directly, start in `src/main.py`.

## Current scope

- This repository is a proof of concept, not a production compliance system.
- The current workflow is scenario-based; there is no user-facing upload interface yet.
- Retrieval currently works over a limited GA4GH PDF corpus.

## Repo guide

Key files:

- `src/main.py` for orchestration
- `src/compliance.py` for prompt construction and JSON parsing
- `src/ingestion/ingest_pdf.py` for PDF parsing and chunking
- `src/ingestion/vector_store.py` for indexing, retrieval, deduplication, and reranking