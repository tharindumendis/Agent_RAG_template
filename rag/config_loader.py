"""
rag/config_loader.py
--------------------
Loads and validates config.yaml into typed dataclasses.
Import and use `load_config()` anywhere in the project.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Dataclass models
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    name: str = "rag-server"
    transport: str = "stdio"  # "stdio" | "sse"
    port: int = 8002
    host: str = "0.0.0.0"


@dataclass
class EmbeddingsConfig:
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class StoreConfig:
    persist_dir: str = ""
    default_collection: str = "default"


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


# ---------------------------------------------------------------------------
# DB Path Fallback
# ---------------------------------------------------------------------------

def get_default_db_path() -> Path:
    """
    Returns a persistent, OS-agnostic path for the ChromaDB store.
    - Windows: C:/Users/<User>/AppData/Local/agent_rag/db
    - macOS: ~/Library/Application Support/agent_rag/db
    - Linux: ~/.local/share/agent_rag/db
    """
    if os.name == "nt":  # Windows
        base_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.name == "posix":
        if "darwin" in sys.platform:  # macOS
            base_dir = Path.home() / "Library" / "Application Support"
        else:  # Linux
            base_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    else:
        base_dir = Path.home()

    final_path = base_dir / "agent_rag" / "vector_store"
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Priority:
    1. config_path (passed to function)
    2. RAG_CONFIG (environment variable)
    3. ./config.yaml (Current Working Directory)
    4. ../config.yaml (Package Root fallback)
    """

    # Define potential locations
    env_path = os.getenv("RAG_CONFIG")
    cwd_path = Path.cwd() / "config.yaml"
    package_root_path = Path(__file__).parent.parent / "config.yaml"

    # Select the first one that exists
    if config_path:
        final_path = Path(config_path)
    elif env_path:
        final_path = Path(env_path)
    elif cwd_path.exists():
        final_path = cwd_path
    else:
        final_path = package_root_path

    # Final check
    if not final_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Checked:\n"
            f"- Explicit path: {config_path}\n"
            f"- Env Var (RAG_CONFIG): {env_path}\n"
            f"- Current Directory: {cwd_path}\n"
            f"- Package Fallback: {package_root_path}\n"
            f"Please ensure a 'config.yaml' exists."
        )

    with open(final_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # --- Server ---
    server_raw = raw.get("server", {})
    server = ServerConfig(
        name=server_raw.get("name", "rag-server"),
        transport=server_raw.get("transport", "stdio"),
        port=int(server_raw.get("port", 8002)),
        host=server_raw.get("host", "0.0.0.0"),
    )

    # --- Embeddings ---
    emb_raw = raw.get("embeddings", {})
    embeddings = EmbeddingsConfig(
        model=emb_raw.get("model", "all-MiniLM-L6-v2"),
        device=emb_raw.get("device", "cpu"),
    )

    # --- Store ---
    store_raw = raw.get("store", {})
    persist_dir = store_raw.get("persist_dir")
    if persist_dir:
        resolved_persist = str(Path(persist_dir).expanduser().resolve())
    else:
        resolved_persist = str(get_default_db_path())

    store = StoreConfig(
        persist_dir=resolved_persist,
        default_collection=store_raw.get("default_collection", "default"),
    )

    # --- Chunking ---
    chunk_raw = raw.get("chunking", {})
    chunking = ChunkingConfig(
        chunk_size=int(chunk_raw.get("chunk_size", 500)),
        chunk_overlap=int(chunk_raw.get("chunk_overlap", 50)),
    )

    return AppConfig(server=server, embeddings=embeddings, store=store, chunking=chunking)
