"""
test_rag.py — Standalone RAG Pipeline Test
-------------------------------------------
Run this script DIRECTLY (no MCP, no agent) to:
  1. Verify all imports work
  2. Download + cache the embedding model
  3. Test chunking on a sample text
  4. Test embedding
  5. Test ChromaDB store (upsert + query + delete)
  6. Test the full ingest -> search pipeline on a real file

Usage:
    .venv\\Scripts\\python.exe test_rag.py
    .venv\\Scripts\\python.exe test_rag.py --file D:/path/to/any.md
    .venv\\Scripts\\python.exe test_rag.py --query "your search query"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure we can import from the same directory
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows so box-drawing / emoji don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ANSI colours ─────────────────────────────────────────────────────────────
def c(code, text): return f"\033[{code}m{text}\033[0m"
OK   = lambda t: print(c("92", f"  [OK] {t}"))
FAIL = lambda t: print(c("31", f"  [FAIL] {t}"))
INFO = lambda t: print(c("90", f"  [..] {t}"))
HEAD = lambda t: print(f"\n{c('1', t)}")
SEP  = lambda: print("-" * 60)


def test_imports() -> bool:
    HEAD("1. Import check")
    try:
        from rag.embedder import embed, prewarm
        OK("rag.embedder")
    except Exception as e:
        FAIL(f"rag.embedder: {e}"); return False
    try:
        from rag.chunker import chunk_text
        OK("rag.chunker")
    except Exception as e:
        FAIL(f"rag.chunker: {e}"); return False
    try:
        from rag.store import upsert_chunks, query_collection, list_collections, delete_collection
        OK("rag.store")
    except Exception as e:
        FAIL(f"rag.store: {e}"); return False
    return True


def test_chunker() -> bool:
    HEAD("2. Chunker")
    from rag.chunker import chunk_text
    sample = "\n\n".join([f"Paragraph {i}: " + "word " * 60 for i in range(10)])
    chunks = chunk_text(sample, chunk_size=300, chunk_overlap=30, source="test")
    INFO(f"Input: {len(sample)} chars → {len(chunks)} chunks")
    if not chunks:
        FAIL("No chunks produced"); return False
    for i, ch in enumerate(chunks[:3]):
        INFO(f"  chunk[{i}]: {len(ch['text'])} chars")
    OK(f"Chunker produced {len(chunks)} chunks")
    return True


def test_embedder() -> bool:
    HEAD("3. Embedder (model download on first run — may take 30-60s)")
    from rag.embedder import embed, prewarm

    INFO("Pre-warming model ...")
    t0 = time.time()
    try:
        prewarm()
        OK(f"Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        FAIL(f"Pre-warm failed: {e}"); return False

    INFO("Embedding 3 test sentences ...")
    texts = ["Hello world", "The quick brown fox", "Agent configuration YAML"]
    t0 = time.time()
    try:
        vecs = embed(texts)
        OK(f"Embedded {len(vecs)} texts in {time.time()-t0:.2f}s | dim={len(vecs[0])}")
    except Exception as e:
        FAIL(f"Embed failed: {e}"); return False

    return True


def test_store() -> bool:
    HEAD("4. ChromaDB store")
    from rag.chunker import chunk_text
    from rag.embedder import embed
    from rag.store import upsert_chunks, query_collection, list_collections, delete_collection

    TEST_COL = "test-col-temp"
    TEST_DIR = str(Path(__file__).parent / "chroma_db")

    texts = ["ChromaDB is a vector database.", "LangGraph builds ReAct agents.",
             "sentence-transformers provides local embeddings.", "MCP is the Model Context Protocol."]
    chunks = [{"text": t, "source": "test", "chunk_idx": i} for i, t in enumerate(texts)]

    INFO("Embedding test chunks ...")
    try:
        embs = embed(texts)
        OK(f"Embeddings ready: {len(embs)} vectors")
    except Exception as e:
        FAIL(f"Embedding: {e}"); return False

    INFO("Upserting to ChromaDB ...")
    try:
        n = upsert_chunks(chunks, embs, collection_name=TEST_COL, persist_dir=TEST_DIR)
        OK(f"Upserted {n} chunks")
    except Exception as e:
        FAIL(f"Upsert: {e}"); return False

    INFO("Querying ...")
    try:
        q_emb = embed(["what is a vector database?"])[0]
        hits = query_collection(q_emb, collection_name=TEST_COL, top_k=2, persist_dir=TEST_DIR)
        if not hits:
            FAIL("Query returned no results"); return False
        OK(f"Query returned {len(hits)} results")
        for h in hits:
            INFO(f"  score={h['score']:.3f} | {h['text'][:60]}")
    except Exception as e:
        FAIL(f"Query: {e}"); return False

    INFO("Listing collections ...")
    try:
        cols = list_collections(persist_dir=TEST_DIR)
        names = [c["name"] for c in cols]
        OK(f"Collections: {names}")
    except Exception as e:
        FAIL(f"List: {e}"); return False

    INFO(f"Deleting test collection '{TEST_COL}' ...")
    try:
        ok = delete_collection(TEST_COL, persist_dir=TEST_DIR)
        OK("Deleted") if ok else FAIL("Delete returned False")
    except Exception as e:
        FAIL(f"Delete: {e}"); return False

    return True


def test_full_pipeline(file_path: str, query: str) -> bool:
    HEAD(f"5. Full pipeline: ingest '{Path(file_path).name}' → search '{query[:50]}'")
    from rag.chunker import chunk_text
    from rag.embedder import embed
    from rag.store import upsert_chunks, query_collection, delete_collection

    TEST_COL = "test-pipeline-temp"
    TEST_DIR = str(Path(__file__).parent / "chroma_db")

    INFO(f"Reading file: {file_path}")
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        INFO(f"File: {len(text)} chars")
    except Exception as e:
        FAIL(f"Read file: {e}"); return False

    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50, source=file_path)
    INFO(f"Chunked into {len(chunks)} chunks")

    INFO("Embedding chunks ...")
    t0 = time.time()
    embs = embed([c["text"] for c in chunks])
    OK(f"Embedded in {time.time()-t0:.2f}s")

    upsert_chunks(chunks, embs, collection_name=TEST_COL, persist_dir=TEST_DIR)
    OK(f"Stored {len(chunks)} chunks")

    INFO(f"Searching: '{query}'")
    q_emb = embed([query])[0]
    hits = query_collection(q_emb, collection_name=TEST_COL, top_k=3, persist_dir=TEST_DIR)
    if not hits:
        FAIL("No search results"); return False

    print()
    for i, h in enumerate(hits, 1):
        print(c("93", f"  Result {i} | score={h['score']:.3f} | {Path(h['source']).name}"))
        print(f"  {h['text'][:200].strip()}")
        print()

    delete_collection(TEST_COL, persist_dir=TEST_DIR)
    OK("Test collection cleaned up")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone RAG test")
    p.add_argument("--file", "-f", default=None,
                   help="File to ingest for pipeline test (default: this script)")
    p.add_argument("--query", "-q", default="How do I configure the model?",
                   help="Search query for pipeline test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SEP()
    print(c("1;96", " RAG Standalone Test Suite"))
    SEP()

    results = []
    results.append(("Imports",  test_imports()))
    results.append(("Chunker",  test_chunker()))
    results.append(("Embedder", test_embedder()))
    results.append(("Store",    test_store()))

    file_to_test = args.file or str(Path(__file__).parent / "server.py")
    results.append(("Pipeline", test_full_pipeline(file_to_test, args.query)))

    HEAD("Results")
    SEP()
    all_ok = True
    for name, ok in results:
        s = c("92", "PASS") if ok else c("31", "FAIL")
        print(f"  {s}  {name}")
        if not ok: all_ok = False
    SEP()
    if all_ok:
        print(c("1;92", "\n  All tests passed! RAG server is ready to use.\n"))
        print(c("90", "  Tip: the embedding model is now cached — first tool call will be fast.\n"))
    else:
        print(c("1;31", "\n  Some tests failed. Fix errors above before running Agent_head.\n"))
    sys.exit(0 if all_ok else 1)
