"""
rag/store.py
------------
ChromaDB wrapper for persistent vector storage, with timing + debug logs.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None


def _safe_name(name: str) -> str:
    """
    Sanitise a ChromaDB collection name.
    Rules: 3-512 chars, [a-zA-Z0-9._-], must start AND end with [a-zA-Z0-9].
    """
    import re
    # Replace anything not allowed with a dash
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    # Strip leading/trailing non-alphanumeric chars
    name = name.strip("-_.")
    # Ensure minimum length
    if len(name) < 3:
        name = f"col-{name}"
    return name[:512]


def get_client(persist_dir: str = "./chroma_db"):
    """Return (and cache) a persistent ChromaDB client."""
    global _client
    if _client is None:
        import chromadb
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.info("[Store] Initialising ChromaDB at '%s' ...", persist_dir)
        t0 = time.time()
        _client = chromadb.PersistentClient(path=persist_dir)
        logger.info("[Store] ChromaDB ready in %.2fs.", time.time() - t0)
    return _client


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str = "default",
    persist_dir: str = "./chroma_db",
) -> int:
    if not chunks:
        return 0

    logger.info("[Store] Upserting %d chunks → collection '%s' ...", len(chunks), collection_name)
    t0 = time.time()
    client = get_client(persist_dir)
    safe = _safe_name(collection_name)
    if safe != collection_name:
        logger.warning("[Store] Collection name sanitised: '%s' → '%s'", collection_name, safe)
    col = client.get_or_create_collection(
        name=safe,
        metadata={"hnsw:space": "cosine"},
    )
    ids       = [f"{c['source']}::{c['chunk_idx']}::{uuid.uuid4().hex[:6]}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in chunks]

    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("[Store] Upserted %d chunks in %.2fs.", len(chunks), time.time() - t0)
    return len(chunks)


def query_collection(
    query_embedding: list[float],
    collection_name: str = "default",
    top_k: int = 5,
    persist_dir: str = "./chroma_db",
) -> list[dict]:
    logger.info("[Store] Querying collection '%s' (top_k=%d) ...", collection_name, top_k)
    t0 = time.time()
    client = get_client(persist_dir)
    safe = _safe_name(collection_name)
    try:
        col = client.get_collection(name=safe)
    except Exception:
        logger.warning("[Store] Collection '%s' not found.", collection_name)
        return []

    count = col.count()
    if count == 0:
        logger.info("[Store] Collection '%s' is empty.", collection_name)
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":      doc,
            "source":    meta.get("source", ""),
            "chunk_idx": meta.get("chunk_idx", 0),
            "score":     round(1.0 - dist, 4),
        })
    logger.info("[Store] Query returned %d results in %.2fs.", len(hits), time.time() - t0)
    return hits


def list_collections(persist_dir: str = "./chroma_db") -> list[dict]:
    client = get_client(persist_dir)
    result = []
    for col in client.list_collections():
        try:
            c = client.get_collection(col.name)
            result.append({"name": col.name, "count": c.count()})
        except Exception:
            result.append({"name": col.name, "count": -1})
    logger.info("[Store] Listed %d collections.", len(result))
    return result


def delete_collection(collection_name: str, persist_dir: str = "./chroma_db") -> bool:
    client = get_client(persist_dir)
    safe = _safe_name(collection_name)
    try:
        client.delete_collection(name=safe)
        logger.info("[Store] Deleted collection '%s'.", collection_name)
        return True
    except Exception as exc:
        logger.warning("[Store] Could not delete '%s': %s", collection_name, exc)
        return False
