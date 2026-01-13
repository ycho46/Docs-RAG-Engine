# rag_server.py
# Local RAG API with Prometheus metrics.
#
# Run from rag/:
#   uvicorn rag_server:app --host 0.0.0.0 --port 8000 --reload

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, Response
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# -----------------------------
# Config
# -----------------------------
INDEX_DIR = Path(os.getenv("INDEX_DIR", "rag_index"))
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "otel_docs_chunks")

TOPK_BM25 = int(os.getenv("RAG_TOPK_BM25", "30"))
TOPK_VEC = int(os.getenv("RAG_TOPK_VEC", "30"))
TOPK_FINAL = int(os.getenv("RAG_TOPK_FINAL", "8"))

EMBED_MODEL_LOCAL = os.getenv("RAG_EMBED_MODEL_LOCAL", "sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "300"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

ENABLE_OTEL = os.getenv("ENABLE_OTEL", "0") == "1"


# -----------------------------
# Prometheus Metrics
# -----------------------------
REQUESTS_TOTAL = Counter("rag_requests_total", "Total RAG requests", ["status"])
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval (BM25 + vector) latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2),
)
LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds",
    "LLM inference latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
)
INFLIGHT = Gauge("rag_inflight_requests", "In-flight RAG requests")


def setup_otel(app: FastAPI) -> None:
    if not ENABLE_OTEL:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({"service.name": "otel-docs-rag"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass


app = FastAPI(title="OTel Docs RAG (Local)")
setup_otel(app)


# -----------------------------
# Load indexes
# -----------------------------
if not BM25_PATH.exists():
    raise SystemExit("bm25.pkl not found. From rag/: python rag_extract_chunks.py && python rag_build_index.py")
if not CHROMA_DIR.exists():
    raise SystemExit("Chroma index not found. From rag/: python rag_extract_chunks.py && python rag_build_index.py")

with BM25_PATH.open("rb") as f:
    bm25_blob = pickle.load(f)
bm25: BM25Okapi = bm25_blob["bm25"]
bm25_ids: List[str] = bm25_blob["ids"]

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_collection(COLLECTION_NAME)

embedder = SentenceTransformer(EMBED_MODEL_LOCAL)


# -----------------------------
# Helpers
# -----------------------------
def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def bm25_search(query: str, k: int) -> List[Tuple[str, float]]:
    scores = bm25.get_scores(tokenize(query))
    idxs = scores.argsort()[-k:][::-1]
    return [(bm25_ids[i], float(scores[i])) for i in idxs]


def vector_search(query: str, k: int) -> List[Tuple[str, float]]:
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype=np.float32)[0].tolist()

    # NOTE: include cannot contain "ids" in Chroma; ids are returned always.
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["distances"],
    )

    hits: List[Tuple[str, float]] = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        hits.append((cid, 1.0 - float(dist)))
    return hits


def merge_and_rank(bm25_hits: List[Tuple[str, float]], vec_hits: List[Tuple[str, float]], topk: int) -> List[str]:
    scores: Dict[str, float] = {}
    bm25_max = max((s for _, s in bm25_hits), default=1.0) or 1.0

    for cid, s in bm25_hits:
        scores[cid] = scores.get(cid, 0.0) + 0.6 * (s / bm25_max)

    for cid, s in vec_hits:
        scores[cid] = scores.get(cid, 0.0) + 0.4 * s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [cid for cid, _ in ranked]


def fetch_chunks(ids: List[str]) -> List[Dict[str, Any]]:
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    out = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        out.append(
            {
                "chunk_id": cid,
                "text": doc or "",
                "url": (meta or {}).get("url", ""),
                "doc_title": (meta or {}).get("doc_title", ""),
                "section": (meta or {}).get("section", ""),
            }
        )
    return out


def call_ollama(system: str, user: str) -> str:
    prompt = f"{system}\n\n{user}"
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        },
        timeout=OLLAMA_TIMEOUT_S,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


# -----------------------------
# API models
# -----------------------------
class AskRequest(BaseModel):
    question: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ollama_model": OLLAMA_MODEL, "embed_model": EMBED_MODEL_LOCAL}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    INFLIGHT.inc()

    try:
        question = req.question.strip()
        if not question:
            REQUESTS_TOTAL.labels(status="bad_request").inc()
            return {"error": "empty question"}

        t0 = time.time()
        bm25_hits = bm25_search(question, TOPK_BM25)
        vec_hits = vector_search(question, TOPK_VEC)
        chosen_ids = merge_and_rank(bm25_hits, vec_hits, TOPK_FINAL)
        chunks = fetch_chunks(chosen_ids)
        RETRIEVAL_LATENCY.observe(time.time() - t0)

        context = "\n\n".join(
            [
                f"[{i+1}] {c['doc_title']} — {c['section']}\nURL: {c['url']}\n{c['text']}"
                for i, c in enumerate(chunks)
            ]
        )

        system_prompt = (
            "You are a technical assistant for OpenTelemetry.\n"
            "Use ONLY the provided documentation context.\n"
            "If the answer is not in the context, say so.\n"
            "Always cite sources as [1], [2], etc."
        )

        user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:"

        t1 = time.time()
        answer = call_ollama(system_prompt, user_prompt)
        LLM_LATENCY.observe(time.time() - t1)

        REQUESTS_TOTAL.labels(status="ok").inc()
        return {"answer": answer, "used_chunks": [c["chunk_id"] for c in chunks]}

    except Exception:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise

    finally:
        REQUEST_LATENCY.observe(time.time() - start)
        INFLIGHT.dec()

