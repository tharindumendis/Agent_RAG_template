"""
server.py — RAG MCP Server
---------------------------
Exposes 4 MCP tools that the orchestrator (Agent_head) can call:

  rag_ingest(source, collection, chunk_size)
  rag_search(query, collection, top_k)
  rag_list_collections()
  rag_delete_collection(collection)

All tool functions are regular synchronous `def` — FastMCP 3.x automatically
runs them in a thread pool. This avoids asyncio event-loop blocking issues
that occur on Windows when async tools call CPU-bound code.

Usage:
    python server.py            # stdio (default)
    python server.py --transport sse --port 8002
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging — always goes to stderr (stdout is for MCP protocol)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from rag.config_loader import load_config

_cfg = load_config()
_EMBED_MODEL   = _cfg.embeddings.model
_EMBED_DEVICE  = _cfg.embeddings.device
_PERSIST_DIR   = _cfg.store.persist_dir
_DEFAULT_COL   = _cfg.store.default_collection
_CHUNK_SIZE    = _cfg.chunking.chunk_size
_CHUNK_OVERLAP = _cfg.chunking.chunk_overlap

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("rag-server")


# ---------------------------------------------------------------------------
# Tool 1: rag_ingest  (sync — FastMCP threads it automatically)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_ingest(
    source: str,
    collection: str = _DEFAULT_COL,
    chunk_size: int = _CHUNK_SIZE,
    metadata: dict | None = None,
) -> str:
    """
    Ingest a document into the RAG knowledge base.

    Args:
        source:     A file path, directory path, or raw text string.
                    - File path  -> reads the file content (.txt .md .py .yaml .json etc.)
                    - Directory  -> recursively reads all text files inside
                    - Raw text   -> used directly (short strings that aren't valid paths)
        collection: Name of the ChromaDB collection to store into. Default: "default"
        chunk_size: Characters per chunk. Default: 500
        metadata:   Metadata to attach to each chunk. Default: None

    Returns:
        Summary of how many chunks were ingested.
    """
    from rag.chunker import chunk_text
    from rag.embedder import embed
    from rag import store

    t_start = time.time()
    logger.info("[rag_ingest] source='%s' collection='%s'", source, collection)

    chunks: list[dict] = []
    src_path = Path(source)

    # ── Directory ────────────────────────────────────────────────────────
    if src_path.is_dir():
        EXTS = {".txt", ".md", ".py", ".yaml", ".yml", ".json", ".toml", ".rst", ".csv", ".log"}
        files = [p for p in src_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        logger.info("[rag_ingest] Found %d files in directory.", len(files))
        if not files:
            return f"No readable files found in directory: {source}"
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                chunks.extend(chunk_text(text, chunk_size=chunk_size,
                                         chunk_overlap=_CHUNK_OVERLAP, source=str(fp)))
            except Exception as exc:
                logger.warning("[rag_ingest] Skipping %s: %s", fp, exc)

    # ── File ─────────────────────────────────────────────────────────────
    elif src_path.is_file():
        try:
            text = src_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text, chunk_size=chunk_size,
                                chunk_overlap=_CHUNK_OVERLAP, source=str(src_path))
            logger.info("[rag_ingest] File chunked into %d chunks.", len(chunks))
        except Exception as exc:
            return f"ERROR reading file '{source}': {exc}"

    # ── Raw text ──────────────────────────────────────────────────────────
    else:
        chunks = chunk_text(source, chunk_size=chunk_size,
                            chunk_overlap=_CHUNK_OVERLAP, source="<inline>")
        logger.info("[rag_ingest] Raw text chunked into %d chunks.", len(chunks))

    if not chunks:
        return "No text content found to ingest (empty source or no matching files)."

    logger.info("[rag_ingest] Embedding %d chunks ...", len(chunks))
    texts = [c["text"] for c in chunks]
    try:
        embeddings = embed(texts, model_name=_EMBED_MODEL, device=_EMBED_DEVICE)
    except Exception as exc:
        logger.exception("[rag_ingest] Embedding failed.")
        return f"ERROR generating embeddings: {exc}"

    try:
        n = store.upsert_chunks(chunks, embeddings,
                                collection_name=collection, persist_dir=_PERSIST_DIR,
                                metadata=metadata)
    except Exception as exc:
        logger.exception("[rag_ingest] Store upsert failed.")
        return f"ERROR storing chunks: {exc}"

    elapsed = time.time() - t_start
    msg = (
        f"Ingested {n} chunks into collection '{collection}' in {elapsed:.1f}s.\n"
        f"Sources: {list({c['source'] for c in chunks})}"
    )
    logger.info("[rag_ingest] Done. %s", msg)
    return msg


# ---------------------------------------------------------------------------
# Tool 2: rag_search  (sync — FastMCP threads it automatically)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_search(
    query: str,
    collection: str = _DEFAULT_COL,
    top_k: int = 5,
    metadata_filter: dict | None = None,
) -> str:
    """
    Search the RAG knowledge base for content relevant to a query.

    Args:
        query:      The search query (natural language).
        collection: ChromaDB collection to search. Default: "default"
        top_k:      Number of top results to return. Default: 5
        metadata_filter: Filter to apply to the search results. Default: None

    Returns:
        Formatted string with the most relevant text chunks and their sources.
    """
    from rag.embedder import embed
    from rag import store

    t_start = time.time()
    logger.info("[rag_search] query='%s...' collection='%s' top_k=%d",
                query[:60], collection, top_k)

    if not query.strip():
        return "ERROR: query cannot be empty."

    try:
        q_embedding = embed([query], model_name=_EMBED_MODEL, device=_EMBED_DEVICE)[0]
    except Exception as exc:
        logger.exception("[rag_search] Embedding failed.")
        return f"ERROR generating query embedding: {exc}"

    try:
        hits = store.query_collection(
            q_embedding, collection_name=collection,
            top_k=top_k, persist_dir=_PERSIST_DIR,
            metadata_filter=metadata_filter,
        )
    except Exception as exc:
        logger.exception("[rag_search] Query failed.")
        return f"ERROR querying collection '{collection}': {exc}"

    if not hits:
        return f"No results found in collection '{collection}'. Try ingesting documents first."

    elapsed = time.time() - t_start
    lines = [f"Top {len(hits)} results from '{collection}' ({elapsed:.2f}s):\n"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"--- Result {i} (score: {hit['score']:.3f}) | source: {hit['source']} ---")
        lines.append(hit["text"])
        lines.append("")

    logger.info("[rag_search] Returned %d results in %.2fs.", len(hits), elapsed)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: rag_list_collections  (sync)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_list_collections() -> str:
    """
    List all RAG collections and how many chunks each contains.
    """
    from rag import store

    try:
        cols = store.list_collections(persist_dir=_PERSIST_DIR)
    except Exception as exc:
        return f"ERROR listing collections: {exc}"

    if not cols:
        return "No collections found. Use rag_ingest to add documents."

    lines = ["RAG Collections:"]
    for col in cols:
        lines.append(f"  - {col['name']}  ({col['count']} chunks)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: rag_delete_collection  (sync)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_delete_collection(collection: str) -> str:
    """
    Delete a RAG collection and all its stored documents.

    Args:
        collection: Name of the collection to delete.
    """
    from rag import store

    try:
        ok = store.delete_collection(collection_name=collection, persist_dir=_PERSIST_DIR)
    except Exception as exc:
        return f"ERROR deleting collection '{collection}': {exc}"

    return f"Collection '{collection}' deleted." if ok else f"Collection '{collection}' not found."


# ---------------------------------------------------------------------------
# Tool 5: rag_delete_items  (sync)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_delete_items(
    collection: str,
    metadata_filter: dict,
) -> str:
    """
    Delete specific items from a RAG collection matching a metadata filter.

    Args:
        collection: Name of the collection to delete from.
        metadata_filter: Dictionary specifying the metadata to filter by.
    """
    from rag import store

    if not metadata_filter:
        return "ERROR: metadata_filter is required. To delete the entire collection, use rag_delete_collection."

    try:
        deleted = store.delete_items(
            collection_name=collection, metadata_filter=metadata_filter, persist_dir=_PERSIST_DIR
        )
    except Exception as exc:
        return f"ERROR deleting items from collection '{collection}': {exc}"

    if deleted >= 0:
        return f"Deleted {deleted} items from collection '{collection}' matching filter {metadata_filter}."
    else:
        return f"Failed to delete items from collection '{collection}' (does it exist?)."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG MCP Server")
    p.add_argument("--transport", choices=["stdio", "sse"], default=_cfg.server.transport)
    p.add_argument("--port", type=int, default=_cfg.server.port)
    p.add_argument("--host", default=_cfg.server.host)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        "RAG Server starting | transport=%s | persist=%s | model=%s",
        args.transport, _PERSIST_DIR, _EMBED_MODEL,
    )

    logger.info("Initializing Chroma DB & Embedder in main thread to avoid deadlocks...")
    try:
        from rag.embedder import prewarm
        prewarm(model_name=_EMBED_MODEL, device=_EMBED_DEVICE)
        from rag.store import get_client
        get_client(persist_dir=_PERSIST_DIR)
        logger.info("Pre-warm complete — RAG server fully ready.")
    except Exception as exc:
        logger.error("Pre-warm failed: %s", exc)

    # -----------------------------------------------------------------------
    # Directory Watcher
    # -----------------------------------------------------------------------
    watcher = None
    if getattr(_cfg, "watch", None) and getattr(_cfg.watch, "enabled", False) and getattr(_cfg.watch, "directories", []):
        try:
            from rag.watcher import DirectoryWatcher
            logger.info("Starting watcher for %d directories...", len(_cfg.watch.directories))
            
            def ingest_cb(path: str, collection: str):
                rag_ingest(source=path, collection=collection, chunk_size=_CHUNK_SIZE)
                
            def delete_cb(collection: str, md_filter: dict):
                rag_delete_items(collection=collection, metadata_filter=md_filter)
                
            watcher = DirectoryWatcher(ingest_cb, delete_cb)
            for directory in _cfg.watch.directories:
                watcher.add_directory(directory)
                
            watcher.start()
        except ImportError as exc:
            logger.warning("Watcher not started (watchdog package missing): %s", exc)
        except Exception as exc:
            logger.error("Failed to initialize directory watcher: %s", exc)

    logger.info("RAG Server MCP transport starting ...")

    try:
        if args.transport == "sse":
            mcp.run(transport="sse", host=args.host, port=args.port)
        else:
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Shutting down RAG Server...")
    finally:
        if watcher:
            watcher.stop()

if __name__ == "__main__":
    main()
