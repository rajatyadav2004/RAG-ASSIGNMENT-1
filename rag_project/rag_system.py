"""
rag_system.py
=============
Hallucination-Aware Retrieval-Augmented Generation (RAG) System
for Technical Question Answering.

Assignment 1 — Generative AI & LLMs Course Project

Features:
  - Loads 30 technical paragraphs from data.txt
  - Chunks text with two strategies: size=200 and size=500
  - Generates sentence-transformer embeddings (all-MiniLM-L6-v2)
  - Indexes into FAISS and ChromaDB vector stores
  - Retrieves top-k semantically similar chunks
  - Computes similarity score and triggers hallucination warning
    if score < threshold
  - Generates answers using a lightweight LLM (GPT-2)
"""

import os
import re
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─── 1. IMPORTS ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import chromadb
from transformers import pipeline

# ─── 2. CONFIGURATION ──────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # Lightweight sentence-transformer model
LLM_MODEL         = "gpt2"               # Lightweight generative LLM
DATA_FILE         = "data.txt"            # Knowledge base file
CHUNK_SIZES       = [200, 500]            # Two chunking configurations (in words)
CHUNK_OVERLAP     = 30                    # Overlap (in words) between chunks
TOP_K             = 3                     # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.30              # Below this → hallucination warning
CHROMA_PERSIST_DIR = "./chroma_db"       # ChromaDB persistence directory

# ─── 3. DATA LOADING ───────────────────────────────────────────────────────────
def load_documents(filepath: str) -> list[str]:
    """Load and split the knowledge base into paragraphs."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on double newlines (paragraphs), strip noise
    paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 80]
    print(f"[Data] Loaded {len(paragraphs)} paragraphs from '{filepath}'")
    return paragraphs


# ─── 4. TEXT CHUNKING ──────────────────────────────────────────────────────────
def chunk_text(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Fixed-size word chunking with overlap.

    Args:
        paragraphs : list of raw document paragraphs
        chunk_size : maximum number of words per chunk
        overlap    : number of words to repeat between consecutive chunks

    Returns:
        list of text chunks
    """
    chunks = []
    for para in paragraphs:
        words = para.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap  # slide with overlap
    print(f"[Chunking] chunk_size={chunk_size}, overlap={overlap} → {len(chunks)} chunks")
    return chunks


# ─── 5. EMBEDDING GENERATION ───────────────────────────────────────────────────
def generate_embeddings(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode chunks into dense vector embeddings."""
    print(f"[Embedding] Encoding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"[Embedding] Shape: {embeddings.shape}")
    return embeddings


# ─── 6. FAISS VECTOR STORE ─────────────────────────────────────────────────────
class FAISSStore:
    """Wraps a FAISS flat L2 index for exact similarity search."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)  # Exact L2 search
        self.chunks: list[str] = []
        self.dim = dim

    def add(self, chunks: list[str], embeddings: np.ndarray):
        """Add documents and their embeddings to the index."""
        # FAISS requires float32
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        print(f"[FAISS] Indexed {self.index.ntotal} vectors (dim={self.dim})")

    def search(self, query_embedding: np.ndarray, top_k: int):
        """
        Retrieve top-k most similar chunks.

        Returns:
            (chunks, distances, similarity_scores)
        """
        q = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(q, top_k)

        results = []
        scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfound
                continue
            # Convert L2 distance to cosine-like similarity (0–1)
            score = float(1 / (1 + dist))
            results.append(self.chunks[idx])
            scores.append(score)
        return results, distances[0].tolist(), scores


# ─── 7. CHROMADB VECTOR STORE ──────────────────────────────────────────────────
class ChromaStore:
    """Wraps ChromaDB for vector storage and retrieval (updated for latest Chroma)."""

    def __init__(self, collection_name: str, persist_dir: str):
        import chromadb

        # ✅ New client (no deprecated Settings)
        self.client = chromadb.Client()

        # Recreate collection for clean runs
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        # ✅ Correct way to create collection
        self.collection = self.client.create_collection(name=collection_name)

        print(f"[Chroma] Collection '{collection_name}' ready")

    def add(self, chunks: list[str], embeddings):
        """Add documents and embeddings to ChromaDB."""
        ids = [f"doc_{i}" for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids
        )

        print(f"[Chroma] Indexed {len(chunks)} documents")

    def search(self, query_embedding, top_k: int):
        """
        Query ChromaDB for top-k similar chunks.

        Returns:
            (chunks, distances, similarity_scores)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count())
        )

        chunks = results["documents"][0]
        distances = results["distances"][0]

        # Convert distance to similarity score
        scores = [float(1 / (1 + d)) for d in distances]

        return chunks, distances, scores


# ─── 8. HALLUCINATION DETECTION ────────────────────────────────────────────────
def check_hallucination(scores: list[float], threshold: float = SIMILARITY_THRESHOLD):
    """
    Determine whether retrieved context is relevant enough to trust.

    Args:
        scores    : list of similarity scores from retrieval
        threshold : minimum acceptable top-1 similarity

    Returns:
        (is_low_confidence: bool, max_score: float)
    """
    if not scores:
        return True, 0.0
    max_score = max(scores)
    is_low_confidence = max_score < threshold
    return is_low_confidence, max_score


# ─── 9. ANSWER GENERATION ──────────────────────────────────────────────────────
def generate_answer(question: str, context_chunks: list[str], generator) -> str:
    """
    Build a prompt from retrieved context and generate an answer with the LLM.

    Args:
        question      : user's question
        context_chunks: top-k retrieved text chunks
        generator     : HuggingFace text-generation pipeline

    Returns:
        generated answer string
    """
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Truncate prompt to fit GPT-2's 1024-token window
    max_input_tokens = 800
    prompt_words = prompt.split()
    if len(prompt_words) > max_input_tokens:
        prompt = " ".join(prompt_words[:max_input_tokens])

    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256,  # GPT-2 EOS token as pad
        truncation=True
    )
    # Extract only the generated portion after "Answer:"
    full_text = output[0]["generated_text"]
    answer = full_text.split("Answer:")[-1].strip()
    return answer


# ─── 10. FULL RAG PIPELINE ─────────────────────────────────────────────────────
def run_rag_pipeline(
    question: str,
    chunks: list[str],
    embeddings: np.ndarray,
    faiss_store: FAISSStore,
    chroma_store: ChromaStore,
    embed_model: SentenceTransformer,
    generator,
    db_backend: str = "faiss",
    verbose: bool = True
):
    """
    End-to-end RAG pipeline:
      1. Embed the query
      2. Retrieve top-k chunks
      3. Check hallucination risk
      4. Generate and return answer

    Args:
        db_backend : "faiss" or "chroma"
    """
    # 1. Embed query
    query_embedding = embed_model.encode([question], convert_to_numpy=True)[0]

    # 2. Retrieve
    if db_backend == "faiss":
        retrieved_chunks, distances, scores = faiss_store.search(query_embedding, TOP_K)
    else:
        retrieved_chunks, distances, scores = chroma_store.search(query_embedding, TOP_K)

    # 3. Hallucination check
    is_low_confidence, max_score = check_hallucination(scores)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Backend : {db_backend.upper()}")
        print(f"Top similarity score: {max_score:.4f} (threshold={SIMILARITY_THRESHOLD})")
        if is_low_confidence:
            print("⚠️  WARNING: Answer may be uncertain due to low context relevance")
        print(f"\nRetrieved Context (top {len(retrieved_chunks)} chunks):")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  [{i+1}] (score={scores[i]:.4f}) {chunk[:120]}...")

    # 4. Generate answer
    answer = generate_answer(question, retrieved_chunks, generator)

    if verbose:
        print(f"\nGenerated Answer:\n{answer}")
        print("="*60)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "similarity_scores": scores,
        "max_score": max_score,
        "is_low_confidence": is_low_confidence,
        "answer": answer,
        "backend": db_backend
    }


# ─── 11. EXPERIMENT: CHUNK SIZE COMPARISON ─────────────────────────────────────
def compare_chunk_sizes(
    paragraphs: list[str],
    embed_model: SentenceTransformer,
    generator,
    test_questions: list[str]
):
    """Compare retrieval quality between chunk_size=200 and chunk_size=500."""
    results = {}

    for chunk_size in CHUNK_SIZES:
        print(f"\n{'#'*60}")
        print(f"# Experiment: chunk_size = {chunk_size}")
        print(f"{'#'*60}")

        chunks = chunk_text(paragraphs, chunk_size=chunk_size, overlap=CHUNK_OVERLAP)
        embeddings = generate_embeddings(chunks, embed_model)

        # Build FAISS index
        dim = embeddings.shape[1]
        store = FAISSStore(dim)
        store.add(chunks, embeddings)

        question_results = []
        for q in test_questions:
            query_embedding = embed_model.encode([q], convert_to_numpy=True)[0]
            retrieved, _, scores = store.search(query_embedding, TOP_K)
            question_results.append({
                "question": q,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": np.mean(scores) if scores else 0.0
            })
            print(f"  Q: {q[:60]}...")
            print(f"     max_score={max(scores):.4f}, avg_score={np.mean(scores):.4f}")

        results[chunk_size] = question_results

    return results


# ─── 12. MAIN ENTRY POINT ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  Hallucination-Aware RAG System — Technical QA")
    print("="*70 + "\n")

    # Load data
    paragraphs = load_documents(DATA_FILE)

    # Load embedding model
    print(f"[Setup] Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load LLM
    print(f"[Setup] Loading LLM: {LLM_MODEL}")
    generator = pipeline("text-generation", model=LLM_MODEL)

    # ── Build index with default chunk size (500) ──────────────────────────────
    chunks_500 = chunk_text(paragraphs, chunk_size=500, overlap=CHUNK_OVERLAP)
    embeddings_500 = generate_embeddings(chunks_500, embed_model)
    dim = embeddings_500.shape[1]

    faiss_store = FAISSStore(dim)
    faiss_store.add(chunks_500, embeddings_500)

    chroma_store = ChromaStore("rag_collection", CHROMA_PERSIST_DIR)
    chroma_store.add(chunks_500, embeddings_500)

    # ── Test Questions ─────────────────────────────────────────────────────────
    test_questions = [
        "What is the difference between a process and a thread in operating systems?",
        "Explain the ACID properties of database transactions.",
        "How does the Transformer architecture use self-attention?",
        "What is RAG and how does it reduce hallucinations in LLMs?",
        "What is LoRA and how does it enable parameter-efficient fine-tuning?",
    ]

    # Out-of-domain question to trigger low-confidence warning
    ood_question = "What is the boiling point of tungsten in Kelvin?"

    print("\n" + "─"*60)
    print("SAMPLE 1: Normal High-Confidence Answer")
    print("─"*60)
    run_rag_pipeline(
        test_questions[0], chunks_500, embeddings_500,
        faiss_store, chroma_store, embed_model, generator,
        db_backend="faiss"
    )

    print("\n" + "─"*60)
    print("SAMPLE 2: Low-Confidence Warning Example")
    print("─"*60)
    run_rag_pipeline(
        ood_question, chunks_500, embeddings_500,
        faiss_store, chroma_store, embed_model, generator,
        db_backend="faiss"
    )

    print("\n" + "─"*60)
    print("SAMPLE 3: FAISS vs Chroma Comparison")
    print("─"*60)
    for backend in ["faiss", "chroma"]:
        run_rag_pipeline(
            test_questions[3], chunks_500, embeddings_500,
            faiss_store, chroma_store, embed_model, generator,
            db_backend=backend
        )

    # ── Chunk Size Comparison Experiment ──────────────────────────────────────
    print("\n" + "─"*60)
    print("EXPERIMENT: Chunk Size 200 vs 500 Comparison")
    print("─"*60)
    compare_chunk_sizes(paragraphs, embed_model, generator, test_questions[:3])

    print("\n[Done] RAG pipeline completed successfully.")


if __name__ == "__main__":
    main()
