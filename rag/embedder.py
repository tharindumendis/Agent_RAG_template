"""
rag/embedder.py
---------------
Pluggable embedder interface with multiple implementations:
  - OnnxEmbedder: chromadb's built-in ONNX embedding (no torch/multiprocessing)
  - OllamaEmbedder: Ollama API-based embedding (lightweight models like nomic-embed-text)

At runtime, the embedder is selected based on config.embeddings.provider.
"""

from __future__ import annotations

import abc
import logging
import os
import requests
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================================================
# Environment Setup (only if NOT using Ollama)
# ============================================================================
# Check config early to avoid loading torch-related packages if using Ollama

_should_suppress_torch_output = True
try:
    # Quick check: if config specifies ollama, skip torch suppression setup
    from pathlib import Path
    
    # Lightweight config check without full YAML parse
    config_path = Path.cwd() / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('provider:'):
                    if 'ollama' in line:
                        _should_suppress_torch_output = False
                        logger.debug("[Embedder] Ollama provider detected — skipping torch env vars")
                    break
except Exception:
    pass  # If anything goes wrong, suppress output anyway (safer default)

# Only set torch-suppression env vars if we might use ONNX
if _should_suppress_torch_output:
    # Prevent any HuggingFace/tqdm output to stdout (MCP pipe safety)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Prevent OpenMP/MKL from spawning threads that could conflict (safe for both)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ============================================================================
# Embedder Interface
# ============================================================================

class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings. Returns list of float vectors."""
        pass

    @abstractmethod
    def prewarm(self) -> None:
        """Pre-load/warm up the embedding model."""
        pass


# ============================================================================
# ONNX Embedder (chromadb built-in)
# ============================================================================

class OnnxEmbedder(BaseEmbedder):
    """
    Uses chromadb's built-in ONNX embedding function.
    
    Replaces sentence-transformers (which causes torch to spawn subprocess workers
    that inherit MCP stdio pipe handles on Windows, causing a permanent deadlock).
    chromadb's DefaultEmbeddingFunction uses onnxruntime directly — 
    no torch, no multiprocessing, no pipe inheritance issues.
    """

    def __init__(self):
        self._ef = None
        self._lock = threading.Lock()

    def _get_onnx_function(self):
        """Return (and cache) chromadb's built-in ONNX embedding function. Thread-safe."""
        if self._ef is None:
            with self._lock:
                if self._ef is None:
                    logger.info("[OnnxEmbedder] Loading ONNX embedding function ...")
                    t0 = time.time()
                    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                    self._ef = DefaultEmbeddingFunction()
                    # Warm up with a dummy call so first real call is fast
                    self._ef(["warmup"])
                    logger.info("[OnnxEmbedder] ONNX embedding ready in %.1fs.", time.time() - t0)
        return self._ef

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings. Returns list of float vectors."""
        if not texts:
            return []
        logger.info("[OnnxEmbedder] Embedding %d texts ...", len(texts))
        t0 = time.time()
        ef = self._get_onnx_function()
        vectors = ef(texts)
        logger.info("[OnnxEmbedder] Embedded %d texts in %.2fs.", len(texts), time.time() - t0)
        return [[float(x) for x in v] for v in vectors]

    def prewarm(self) -> None:
        """Pre-load the ONNX embedding model."""
        logger.info("[OnnxEmbedder] Pre-warming ONNX embedding ...")
        self._get_onnx_function()
        logger.info("[OnnxEmbedder] Pre-warm complete.")


# ============================================================================
# Ollama Embedder
# ============================================================================

class OllamaEmbedder(BaseEmbedder):
    """
    Uses Ollama API for lightweight embedding models.
    
    Supports models like:
      - nomic-embed-text (lightweight, fast)
      - all-minilm:latest
      - mxbai-embed-large
    
    Requires Ollama server running at base_url.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._ready = False
        self._lock = threading.Lock()

    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is available and model is loaded."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            logger.error(
                "[OllamaEmbedder] Ollama server not available at %s. "
                "Make sure Ollama is running: ollama serve",
                self.base_url
            )
            return False

    def _ensure_model_pulled(self) -> bool:
        """Ensure the model is pulled (downloaded) in Ollama."""
        try:
            logger.info("[OllamaEmbedder] Pulling model '%s' from Ollama ...", self.model)
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=120,  # model pull can take time
            )
            if response.status_code != 200:
                logger.error("[OllamaEmbedder] Failed to pull model: %s", response.text)
                return False
            logger.info("[OllamaEmbedder] Model '%s' ready.", self.model)
            return True
        except requests.RequestException as e:
            logger.error("[OllamaEmbedder] Error pulling model: %s", e)
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings via Ollama API."""
        if not texts:
            return []

        if not self._ready:
            with self._lock:
                if not self._ready:
                    if not self._check_ollama_available():
                        raise RuntimeError(
                            f"Ollama not available at {self.base_url}. "
                            "Start Ollama with: ollama serve"
                        )
                    if not self._ensure_model_pulled():
                        raise RuntimeError(f"Failed to pull Ollama model: {self.model}")
                    self._ready = True

        logger.info("[OllamaEmbedder] Embedding %d texts with model '%s' ...", len(texts), self.model)
        t0 = time.time()

        embeddings = []
        try:
            for text in texts:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )
                if response.status_code != 200:
                    logger.error("[OllamaEmbedder] Embedding failed: %s", response.text)
                    raise RuntimeError(f"Ollama API error: {response.text}")
                data = response.json()
                embeddings.append(data.get("embedding", []))

            logger.info("[OllamaEmbedder] Embedded %d texts in %.2fs.", len(texts), time.time() - t0)
            return embeddings
        except requests.RequestException as e:
            logger.error("[OllamaEmbedder] Request error: %s", e)
            raise

    def prewarm(self) -> None:
        """Ensure Ollama is available and model is pulled."""
        logger.info("[OllamaEmbedder] Pre-warming Ollama embeddings ...")
        if not self._check_ollama_available():
            raise RuntimeError(
                f"Ollama not available at {self.base_url}. "
                "Start Ollama with: ollama serve"
            )
        if not self._ensure_model_pulled():
            raise RuntimeError(f"Failed to pull Ollama model: {self.model}")
        self._ready = True
        logger.info("[OllamaEmbedder] Pre-warm complete.")


# ============================================================================
# Factory & Global State
# ============================================================================

_embedder = None
_embedder_lock = threading.Lock()


def get_embedder(config=None) -> BaseEmbedder:
    """
    Get or create the embedder based on config.
    
    If config is None, imports and uses the global config.
    Returns the appropriate embedder (ONNX or Ollama) based on config.embeddings.provider.
    """
    global _embedder

    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        # Load config if not provided
        if config is None:
            from rag.config_loader import load_config
            config = load_config()

        provider = getattr(config.embeddings, 'provider', 'onnx').lower()

        if provider == 'ollama':
            base_url = getattr(config.embeddings, 'ollama_base_url', 'http://localhost:11434')
            model = getattr(config.embeddings, 'ollama_model', 'nomic-embed-text')
            logger.info("[Embedder] Using Ollama provider: model='%s', url='%s'", model, base_url)
            _embedder = OllamaEmbedder(base_url=base_url, model=model)
        else:
            logger.info("[Embedder] Using ONNX provider (default)")
            _embedder = OnnxEmbedder()

        return _embedder


# ============================================================================
# Public API (backwards compatibility)
# ============================================================================

def embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",  # kept for API compat
    device: str = "cpu",                     # kept for API compat
) -> list[list[float]]:
    """
    Embed a list of text strings. Uses configured embedder at runtime.
    
    model_name and device are ignored—use config.yaml to select embedder.
    Returns list of float vectors.
    """
    embedder = get_embedder()
    return embedder.embed(texts)


def prewarm(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
    """
    Pre-load the embedding model. Uses configured embedder at runtime.
    
    model_name and device are ignored—use config.yaml to select embedder.
    """
    embedder = get_embedder()
    embedder.prewarm()
