"""
rag/embedder.py
---------------
Embedding using chromadb's built-in ONNX embedding function.

Replaces sentence-transformers (which causes torch to spawn subprocess workers
that inherit MCP stdio pipe handles on Windows, causing a permanent deadlock).

chromadb's DefaultEmbeddingFunction uses onnxruntime directly — 
no torch, no multiprocessing, no pipe inheritance issues.
"""

from __future__ import annotations

import logging
import os
import threading
import time

# Prevent any HuggingFace/tqdm output to stdout (MCP pipe safety)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
# Prevent OpenMP/MKL from spawning threads that could conflict
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

_ef = None
_ef_lock = threading.Lock()


def get_embedding_function():
    """Return (and cache) chromadb's built-in ONNX embedding function. Thread-safe."""
    global _ef
    if _ef is None:
        with _ef_lock:
            if _ef is None:
                logger.info("[Embedder] Loading ONNX embedding function ...")
                t0 = time.time()
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                _ef = DefaultEmbeddingFunction()
                # Warm up with a dummy call so first real call is fast
                _ef(["warmup"])
                logger.info("[Embedder] ONNX embedding ready in %.1fs.", time.time() - t0)
    return _ef


def embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",  # kept for API compat, ignored
    device: str = "cpu",                     # kept for API compat, ignored
) -> list[list[float]]:
    """Embed a list of text strings. Returns list of float vectors."""
    if not texts:
        return []
    logger.info("[Embedder] Embedding %d texts ...", len(texts))
    t0 = time.time()
    ef = get_embedding_function()
    vectors = ef(texts)
    logger.info("[Embedder] Embedded %d texts in %.2fs.", len(texts), time.time() - t0)
    return [[float(x) for x in v] for v in vectors]


def prewarm(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
    """Pre-load the ONNX embedding model."""
    logger.info("[Embedder] Pre-warming ONNX embedding ...")
    get_embedding_function()
    logger.info("[Embedder] Pre-warm complete.")
