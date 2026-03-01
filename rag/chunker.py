"""
rag/chunker.py
--------------
Simple text chunker. Splits on paragraphs/newlines,
respects chunk_size with a configurable overlap.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str = "",
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns a list of dicts:
        {"text": str, "source": str, "chunk_idx": int}
    """
    if not text or not text.strip():
        return []

    # Split on double-newlines (paragraphs) first, then re-join into chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[dict] = []
    current = ""
    idx = 0

    for para in paragraphs:
        # If adding this paragraph keeps us under chunk_size, append it
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            # Flush current chunk
            if current:
                chunks.append({"text": current, "source": source, "chunk_idx": idx})
                idx += 1
                # Carry overlap: keep last `chunk_overlap` chars
                current = current[-chunk_overlap:].strip() if chunk_overlap else ""
            # If a single paragraph is larger than chunk_size, hard-split it
            while len(para) > chunk_size:
                chunks.append({
                    "text": para[:chunk_size],
                    "source": source,
                    "chunk_idx": idx,
                })
                idx += 1
                para = para[chunk_size - chunk_overlap:]
            current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append({"text": current, "source": source, "chunk_idx": idx})

    return chunks
