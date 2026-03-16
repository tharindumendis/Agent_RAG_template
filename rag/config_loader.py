"""
rag/config_loader.py
--------------------
Loads and validates config.yaml into typed dataclasses.
Import and use `load_config()` anywhere in the project.

Config resolution priority
---------------------------
1. Explicit `config_path` argument passed in code
2. RAG_CONFIG environment variable
3. OS user-config dir  (created from bundled default on first run)
       Windows : %LOCALAPPDATA%\\agent_rag\\config.yaml
       macOS   : ~/Library/Application Support/agent_rag/config.yaml
       Linux   : $XDG_CONFIG_HOME/agent_rag/config.yaml  (~/.config/…)
4. <cwd>/config.yaml
5. Bundled package default  (Agent_rag/config.yaml next to server.py)

On first run (priority-3 path missing) the bundled default is copied there
so users can easily edit it without touching the package installation.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

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
class WatchConfig:
    enabled: bool = False
    directories: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_path: str = "logs/rag_server.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# OS-specific paths
# ---------------------------------------------------------------------------


def get_app_config_dir() -> Path:
    """
    Returns the OS-specific user-editable config directory for agent_rag.

    - Windows : %LOCALAPPDATA%\\agent_rag
    - macOS   : ~/Library/Application Support/agent_rag
    - Linux   : $XDG_CONFIG_HOME/agent_rag  (default ~/.config/agent_rag)
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux / other POSIX
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))

    config_dir = base / "agent_rag"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_default_db_path() -> Path:
    """
    Returns a persistent, OS-agnostic path for the ChromaDB vector store.

    - Windows : %LOCALAPPDATA%\\agent_rag\\vector_store
    - macOS   : ~/Library/Application Support/agent_rag/vector_store
    - Linux   : $XDG_DATA_HOME/agent_rag/vector_store  (~/.local/share/…)
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    db_path = base / "agent_rag" / "vector_store"
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path


# ---------------------------------------------------------------------------
# Bootstrap — copy bundled default config to OS user-config dir on first run
# ---------------------------------------------------------------------------

# The bundled default sits next to server.py (package root)
_PACKAGE_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


def bootstrap_config() -> Path:
    """
    Ensures a user-editable config.yaml exists in the OS config directory.

    - If it already exists  → does nothing, returns the existing path.
    - If it doesn't exist   → copies the bundled default there and logs a
                               message so the user knows where to find it.

    Returns the path to the user config file (whether new or pre-existing).
    """
    user_config = get_app_config_dir() / "config.yaml"

    if user_config.exists():
        return user_config

    # First run — copy the bundled default
    if _PACKAGE_DEFAULT_CONFIG.exists():
        shutil.copy2(_PACKAGE_DEFAULT_CONFIG, user_config)
        logger.info(
            "[agent_rag] First-run bootstrap: config copied to %s\n"
            "            Edit that file to customise your RAG server settings.",
            user_config,
        )
    else:
        logger.warning(
            "[agent_rag] Bundled default config not found at %s; "
            "skipping bootstrap. User config path: %s",
            _PACKAGE_DEFAULT_CONFIG,
            user_config,
        )

    return user_config


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Load the RAG server config.  Resolution priority:

    1. ``config_path``  — explicit path passed by the caller
    2. ``RAG_CONFIG``   — environment variable (used when Agent_head spawns
                          the server with a custom override config)
    3. OS user-config   — bootstrapped on first run from the bundled default
       ``%LOCALAPPDATA%\\agent_rag\\config.yaml``  (Windows)
       ``~/Library/Application Support/agent_rag/config.yaml``  (macOS)
       ``~/.config/agent_rag/config.yaml``  (Linux)
    4. ``<cwd>/config.yaml``           — dev / monorepo convenience
    5. ``<package_root>/config.yaml``  — bundled package fallback
    """
    env_path = os.getenv("RAG_CONFIG")
    user_config_path = bootstrap_config()
    cwd_path = Path.cwd() / "config.yaml"
    package_root_path = _PACKAGE_DEFAULT_CONFIG

    if config_path:
        final_path = Path(config_path)
    elif env_path:
        final_path = Path(env_path)
    elif user_config_path.exists():
        final_path = user_config_path
    elif cwd_path.exists():
        final_path = cwd_path
    else:
        final_path = package_root_path

    if not final_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Checked:\n"
            f"  1. Explicit path       : {config_path}\n"
            f"  2. Env var RAG_CONFIG  : {env_path}\n"
            f"  3. OS user-config      : {user_config_path}\n"
            f"  4. CWD                 : {cwd_path}\n"
            f"  5. Package default     : {package_root_path}\n"
            f"Please ensure a 'config.yaml' exists at one of the above locations."
        )

    logger.info("[agent_rag] Using config: %s", final_path.resolve())

    with open(final_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve relative paths in the config relative to the config file's own
    # directory, not the CWD — avoids surprises when launched from different dirs.
    config_dir = final_path.resolve().parent

    def resolve_path(p: str) -> str:
        """Make a relative path absolute, anchored to the config file's dir."""
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return str(path)

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
    persist_dir = store_raw.get("persist_dir", "")
    if persist_dir:
        resolved_persist = resolve_path(persist_dir)
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

    # --- Watch ---
    watch_raw = raw.get("watch", {})
    watch = WatchConfig(
        enabled=watch_raw.get("enabled", False),
        directories=watch_raw.get("directories", []),
    )

    # --- Logging ---
    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        file_path=resolve_path(logging_raw.get("file_path", "./logs/rag_server.log")),
        max_file_size=int(logging_raw.get("max_file_size", 10485760)),
        backup_count=int(logging_raw.get("backup_count", 5)),
    )

    return AppConfig(server=server, embeddings=embeddings, store=store, chunking=chunking, watch=watch, logging=logging_config)
