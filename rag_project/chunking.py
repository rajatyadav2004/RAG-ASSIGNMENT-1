"""
chunking.py
===========
Text chunking module — supports multiple chunk sizes for comparison.
"""


def chunk_text(paragraphs: list, chunk_size: int = 500, overlap: int = 30) -> list:
    """
    Chunk paragraphs into fixed-size word chunks with overlap.

    Args:
        paragraphs: List of paragraph strings
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks

    Returns:
        List of chunk strings
    """
    chunks = []
    for para in paragraphs:
        words = para.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks


def multi_chunk_comparison(paragraphs: list) -> dict:
    """
    Generate chunks at multiple sizes for comparison.

    Returns:
        Dict with chunk_size as key and list of chunks as value
    """
    sizes = [100, 200, 350, 500]
    results = {}
    for size in sizes:
        chunks = chunk_text(paragraphs, chunk_size=size, overlap=20)
        results[size] = {
            "chunks": chunks,
            "count": len(chunks),
            "avg_words": round(sum(len(c.split()) for c in chunks) / max(len(chunks), 1), 1)
        }
    return results
