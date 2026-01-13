# rag_build_index.py
# Build BM25 + Chroma vector index from chunks.jsonl
#
# Run from rag/:
#   python rag_build_index.py
#
# Config (optional):
#   CHUNKS_IN=chunks.jsonl
#   INDEX_DIR=rag_index
#   COLLECTION_NAME=otel_docs_chunks
#   EMBED_MODEL_LOCAL=sentence-transformers/all-MiniLM-L6-v2

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_IN = Path(os.getenv("CHUNKS_IN", "chunks.jsonl"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "rag_index"))
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "otel_docs_chunks")
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL", "sentence-transformers/all-MiniLM-L6-v2")

BATCH = int(os.getenv("EMBED_BATCH", "128"))


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def main() -> None:
    if not CHUNKS_IN.exists():
        raise SystemExit(f"{CHUNKS_IN} not found. Run: python rag_extract_chunks.py")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Load chunks
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    with CHUNKS_IN.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["id"])
            docs.append(rec["text"])
            metas.append(
                {
                    "url": rec.get("url", ""),
                    "doc_title": rec.get("doc_title", ""),
                    "section": rec.get("section", ""),
                }
            )

    if not docs:
        raise SystemExit("No chunks loaded. Check your otel-docs/content is not empty.")

    print(f"Loaded {len(docs)} chunks. Building BM25…")
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"✅ Wrote {BM25_PATH.resolve()}")

    print("Building Chroma vector index…")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Reset collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(COLLECTION_NAME)

    embedder = SentenceTransformer(EMBED_MODEL_LOCAL)

    # Embed in batches
    for start in range(0, len(docs), BATCH):
        end = min(start + BATCH, len(docs))
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]

        embs = embedder.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32).tolist()

        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)
        print(f"  added {end}/{len(docs)}")

    print(f"✅ Chroma persisted at {CHROMA_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()

