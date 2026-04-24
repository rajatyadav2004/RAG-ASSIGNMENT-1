"""
retrieval.py
============
Vector database retrieval module — supports FAISS and ChromaDB.
"""

import numpy as np
import faiss
import chromadb
from embedding import generate_embeddings, embed_query


# ── FAISS ──────────────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list, model_name: str) -> tuple:
    """Build a FAISS index from chunks."""
    embeddings = generate_embeddings(chunks, model_name)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def search_faiss(index, query: str, chunks: list, model_name: str, top_k: int = 3) -> tuple:
    """Search FAISS index and return top_k chunks with scores."""
    q_emb = embed_query(query, model_name)
    D, I = index.search(np.array([q_emb]), top_k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    scores = [float(1 / (1 + d)) for d in D[0]]
    return retrieved, scores


# ── CHROMA ─────────────────────────────────────────────────────────────────────

def build_chroma_collection(chunks: list, model_name: str, collection_name: str = "rag_col"):
    """Build a ChromaDB collection from chunks."""
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)
    embeddings = generate_embeddings(chunks, model_name)

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    return collection


def search_chroma(collection, query: str, model_name: str, top_k: int = 3) -> tuple:
    """Search ChromaDB collection and return top_k chunks with scores."""
    q_emb = embed_query(query, model_name)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k
    )
    retrieved = results["documents"][0]
    distances = results["distances"][0]
    scores = [float(1 / (1 + d)) for d in distances]
    return retrieved, scores
