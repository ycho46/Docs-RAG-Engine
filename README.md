# Docs RAG Engine (Local, Observable)

A small, self-contained **document-based LLM (RAG) engine** designed to index and answer questions over **any Markdown documentation corpus**.

This project uses OpenTelemetry documentation as an **example corpus** to validate the system.

---

## What This Project Is

This repository contains a **portable RAG engine** that can:

- Ingest Markdown documentation
- Chunk and index content (BM25 + vector search)
- Run **local LLM inference** via Ollama
- Expose **Prometheus metrics** for cost, latency, and capacity analysis
- Integrate with OpenTelemetry / Chronosphere for tracing

It is intentionally:

- corpus-agnostic
- API-free (no external LLM services)
- observability-first

---

## What This Project Is NOT

- ❌ Not an OpenTelemetry documentation repo
- ❌ Not a chatbot product
- ❌ Not tied to a specific vendor or model
- ❌ Not production-hardened (by design)

---

## Intended Use Cases

- Learning how to build **observable LLM systems**
- Experimenting with **RAG architectures**
- Cost and latency modeling for local inference
- Indexing:
  - technical documentation
  - runbooks
  - RFCs
  - internal wikis
  - design docs
  - blog archives

---

## High-Level Architecture

```
Documentation Corpus (Markdown)
            │
            ▼
rag_extract_chunks.py
            │   → chunks.jsonl
            ▼
rag_build_index.py
            │   → BM25 + Vector Indexes
            ▼
rag_server.py (FastAPI)
            ├─ Hybrid retrieval
            ├─ Local LLM inference (Ollama)
            ├─ Prometheus metrics (/metrics)
            ▼
Clients (curl / UI)
```

---

## Repository Layout

```
rag/
├── rag_extract_chunks.py
├── rag_build_index.py
├── rag_server.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── .gitignore
├── chunks.jsonl        # generated (ignored)
├── rag_index/          # generated (ignored)
└── .venv/              # local venv (ignored)
```

---

## Corpus Handling

The documentation corpus is **not committed** to this repository.

It is provided at runtime:

```bash
CORPUS_DIR=/path/to/docs
```

Examples:

```bash
CORPUS_DIR=../otel-docs
CORPUS_DIR=/mnt/runbooks
CORPUS_DIR=/data/wiki
```

---

## API Endpoints

| Endpoint     | Description                 |
| ------------ | --------------------------- |
| GET /health  | Service health + model info |
| POST /ask    | Ask a question against docs |
| GET /metrics | Prometheus metrics          |

---

## Prometheus Metrics

- `rag_requests_total` — total request count
- `rag_request_duration_seconds` — end-to-end latency
- `rag_retrieval_duration_seconds` — retrieval time
- `rag_llm_duration_seconds` — inference time (cost proxy)
- `rag_inflight_requests` — concurrent requests

---

## Example Workflow

```bash
export CORPUS_DIR=/path/to/docs
python rag_extract_chunks.py
python rag_build_index.py
uvicorn rag_server:app --port 8000
```

---

## Next Steps

- Add OpenTelemetry tracing
- Add token estimation metrics
- Add GPU utilization metrics
- Build cost-per-request models

---

This project is intended as a **learning and reference implementation** for observable, cost-aware document LLM systems.
