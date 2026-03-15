"""
rag/watcher.py
--------------
Monitors configured directories for file changes and auto-ingests them into ChromaDB.
"""

import time
import logging
import threading
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None  # type: ignore[assignment,misc]
    # Fallback so RagEventHandler can always be defined at module level
    class FileSystemEventHandler:  # type: ignore[no-redef]
        """No-op stub used when watchdog is not installed."""
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)

class RagEventHandler(FileSystemEventHandler):
    def __init__(self, collection_name, ingest_callback, delete_callback):
        super().__init__()
        self.collection_name = collection_name
        self.ingest_callback = ingest_callback
        self.delete_callback = delete_callback
        self.debounce_timers = {}
        self.debounce_seconds = 2.0
        self.supported_exts = {".txt", ".md", ".py", ".yaml", ".yml", ".json", ".toml", ".rst", ".csv", ".log"}

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.supported_exts

    def on_created(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            self._handle_change(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            self._handle_change(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            logger.info("[Watcher] File deleted: %s", event.src_path)
            self._cancel_timer(event.src_path)
            try:
                # Use resolved absolute path for the source metadata
                src_path = str(Path(event.src_path).resolve())
                self.delete_callback(self.collection_name, {"source": src_path})
            except Exception as e:
                logger.error("[Watcher] Failed to delete %s: %s", event.src_path, e)

    def _cancel_timer(self, path):
        if path in self.debounce_timers:
            self.debounce_timers[path].cancel()
            del self.debounce_timers[path]

    def _handle_change(self, path):
        self._cancel_timer(path)
        timer = threading.Timer(self.debounce_seconds, self._process_file, args=[path])
        self.debounce_timers[path] = timer
        timer.start()

    def _process_file(self, path):
        self._cancel_timer(path)
        logger.info("[Watcher] File changed, ingesting: %s", path)
        try:
            src_path = str(Path(path).resolve())
            # 1. Delete existing chunks for this file
            self.delete_callback(self.collection_name, {"source": src_path})
            # 2. Re-ingest the file
            self.ingest_callback(src_path, self.collection_name)
        except Exception as e:
            logger.error("[Watcher] Failed to process %s: %s", path, e)


class DirectoryWatcher:
    def __init__(self, ingest_callback, delete_callback):
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog package is not installed.")
        self.observer = Observer()
        self.ingest_callback = ingest_callback
        self.delete_callback = delete_callback
        self.watches = []
        self.supported_exts = {".txt", ".md", ".py", ".yaml", ".yml", ".json", ".toml", ".rst", ".csv", ".log"}

    def add_directory(self, path: str):
        p = Path(path).resolve()
        if not p.exists():
            logger.info("[Watcher] Directory does not exist, creating it: %s", p)
            p.mkdir(parents=True, exist_ok=True)
            
        collection_name = p.name
        handler = RagEventHandler(collection_name, self.ingest_callback, self.delete_callback)
        watch = self.observer.schedule(handler, str(p), recursive=True)
        self.watches.append(watch)
        logger.info("[Watcher] Watching directory '%s' into collection '%s'", p, collection_name)
        
        # Initial sync: ingest files on startup, removing older chunks first
        self._initial_sync(p, collection_name)

    def _initial_sync(self, directory: Path, collection_name: str):
        logger.info("[Watcher] Performing initial sync for '%s'", directory)
        try:
            files_to_sync = [fp for fp in directory.rglob("*") if fp.is_file() and fp.suffix.lower() in self.supported_exts]
            if not files_to_sync:
                logger.info("[Watcher] No supported files found for initial sync in '%s'", directory)
                return

            for fp in files_to_sync:
                # Use resolved absolute path matching what rag_ingest uses
                src_path = str(fp.resolve())
                self.delete_callback(collection_name, {"source": src_path})
                # Re-ingest individually or bulk ingest? Bulk ingest is easier by calling on directory, 
                # but since we already deleted per-file, calling ingest on the directory is safe and faster.
            
            # Re-ingest the directory
            self.ingest_callback(str(directory), collection_name)
            logger.info("[Watcher] Initial sync complete for '%s'", directory)
        except Exception as e:
            logger.error("[Watcher] Init sync failed for %s: %s", directory, e)

    def start(self):
        if self.watches:
            self.observer.start()
            logger.info("[Watcher] Started monitoring %d directories.", len(self.watches))

    def stop(self):
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("[Watcher] Stopped monitoring directories.")
