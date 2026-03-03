"""
rag/store.py
------------
ChromaDB wrapper for persistent vector storage, with timing + debug logs.
"""

from __future__ import annotations

import logging
import time
import uuid
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()

# How many times to retry a query when ChromaDB throws an internal HNSW error
# ("Error finding id") caused by a freshly written record not yet being
# consistently indexed.  Each retry waits _QUERY_RETRY_DELAY seconds.
_QUERY_MAX_RETRIES   = 3
_QUERY_RETRY_DELAY   = 0.15   # seconds


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
    """
    Return (and cache) a persistent ChromaDB client.

    Thread-safe via double-checked locking: only one thread creates the client.
    Must be called while the caller already holds _client_lock (or during
    server startup before threads are running).
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # re-check inside the lock
                import chromadb
                from chromadb.config import Settings
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                logger.info("[Store] Initialising ChromaDB at '%s' ...", persist_dir)
                t0 = time.time()
                _client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                logger.info("[Store] ChromaDB ready in %.2fs.", time.time() - t0)
    return _client


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str = "default",
    persist_dir: str = "./chroma_db",
    metadata: dict | None = None,
) -> int:
    if not chunks:
        return 0

    logger.info("[Store] Upserting %d chunks → collection '%s' ...", len(chunks), collection_name)
    t0 = time.time()
    
    with _client_lock:
        client = get_client(persist_dir)
        safe = _safe_name(collection_name)
        if safe != collection_name:
            logger.warning("[Store] Collection name sanitised: '%s' → '%s'", collection_name, safe)
        col = client.get_or_create_collection(
            name=safe,
            metadata={"hnsw:space": "cosine"},
        )

        ids = []
        docs = []
        embs = []
        metas = []

        for i, (chk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            docs.append(chk["text"])
            embs.append(emb)

            meta = {
                "source": chk["source"],
                "chunk_idx": chk["chunk_idx"],
            }
            if metadata:
                meta.update(metadata)
            metas.append(meta)

        col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        
    logger.info("[Store] Upserted %d chunks in %.2fs.", len(chunks), time.time() - t0)
    return len(chunks)


def query_collection(
    query_embedding: list[float],
    collection_name: str = "default",
    top_k: int = 5,
    persist_dir: str = "./chroma_db",
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Query the ChromaDB collection.

    Retries up to _QUERY_MAX_RETRIES times when ChromaDB throws an internal
    'Error finding id' error.  This error is caused by the HNSW in-memory
    index returning an ID that was concurrently added but whose SQLite row
    has not yet been fully flushed.  A short sleep lets the write complete.
    """
    logger.info("[Store] Querying collection '%s' (top_k=%d) ...", collection_name, top_k)
    t0 = time.time()

    last_exc: Exception | None = None
    for attempt in range(_QUERY_MAX_RETRIES + 1):
        if attempt > 0:
            logger.warning(
                "[Store] Retry %d/%d for query on '%s' after error: %s",
                attempt, _QUERY_MAX_RETRIES, collection_name, last_exc,
            )
            time.sleep(_QUERY_RETRY_DELAY * attempt)  # back-off slightly

        try:
            with _client_lock:
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

                kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, count),
                    "include": ["documents", "metadatas", "distances"],
                }
                if metadata_filter:
                    kwargs["where"] = metadata_filter

                results = col.query(**kwargs)

            # Build hit list outside the lock (pure Python)
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
                    "metadata":  meta,
                })
            logger.info("[Store] Query returned %d results in %.2fs.", len(hits), time.time() - t0)
            return hits

        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            # Only retry on the specific ChromaDB HNSW consistency error
            if "error finding id" in err_str or "internal error" in err_str:
                continue   # will retry
            # Any other error: raise immediately
            raise

    # All retries exhausted
    logger.error(
        "[Store] query_collection failed after %d retries: %s",
        _QUERY_MAX_RETRIES, last_exc,
    )
    raise last_exc  # type: ignore[misc]


def list_collections(persist_dir: str = "./chroma_db") -> list[dict]:
    with _client_lock:
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
    safe = _safe_name(collection_name)
    with _client_lock:
        client = get_client(persist_dir)
        try:
            client.delete_collection(name=safe)
            logger.info("[Store] Deleted collection '%s'.", safe)
            return True
        except Exception as exc: # Changed ValueError to Exception to match original broader catch
            logger.warning("[Store] Could not delete '%s': %s", collection_name, exc)
            return False

def delete_items(collection_name: str, metadata_filter: dict, persist_dir: str = "./chroma_db") -> int:
    safe = _safe_name(collection_name)
    with _client_lock:
        client = get_client(persist_dir)
        try:
            col = client.get_collection(name=safe)
        except Exception:
            logger.warning("[Store] Collection '%s' not found for item deletion.", collection_name)
            return 0
        
        initial_count = col.count()
        try:
            col.delete(where=metadata_filter)
            deleted = initial_count - col.count()
            logger.info("[Store] Deleted %d items from collection '%s' matching %s.", deleted, collection_name, metadata_filter)
            return deleted
        except Exception as e:
            logger.error("[Store] Error deleting items from %s: %s", safe, e)
            return 0
