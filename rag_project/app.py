"""
app.py
======
Main Streamlit Dashboard — RAG System with:
- Multiple Embedding Models
- Multiple Vector Databases (FAISS + ChromaDB)
- Multiple LLMs
- Chunk Size Comparison
- Full Evaluation Metrics (Faithfulness, Relevancy, Confidence)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Local Modules ──────────────────────────────────────────────────────────────
from ingestion import load_documents, extract_paragraphs
from chunking import chunk_text, multi_chunk_comparison
from embedding import EMBEDDING_MODELS, generate_embeddings, embed_query
from retrieval import (
    build_faiss_index, search_faiss,
    build_chroma_collection, search_chroma
)
from generation import LLM_MODELS, generate_answer, generate_all_models
from evaluation import full_evaluation, keyword_overlap

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Dashboard", layout="wide", page_icon="🤖")

st.title("🤖 RAG AI Assistant")
st.markdown("### Retrieval-Augmented Generation — Multi-Config Experimental System")

# ── Sidebar Settings ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("#### 📦 Embedding Model")
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    list(EMBEDDING_MODELS.keys()),
    help="Model used to convert text to vectors"
)
emb_info = EMBEDDING_MODELS[embedding_model]
st.sidebar.caption(f"Dimension: {emb_info['dimension']} | {emb_info['description']}")

st.sidebar.markdown("#### 🗄️ Vector Database")
backend = st.sidebar.selectbox("Select Vector DB", ["FAISS", "ChromaDB"])
st.sidebar.caption(
    "FAISS: Fast, in-memory, store+search only\n"
    "ChromaDB: Full DB — store, manage, search + persistent"
)

st.sidebar.markdown("#### ✂️ Chunk Size")
chunk_size = st.sidebar.slider("Chunk Size (words)", 100, 500, 350, step=50)
overlap = st.sidebar.slider("Overlap (words)", 10, 50, 20)

st.sidebar.markdown("#### 🧠 LLM Model")
llm_model = st.sidebar.selectbox(
    "Select LLM",
    list(LLM_MODELS.keys()),
    help="Model used to generate final answer"
)
llm_info = LLM_MODELS[llm_model]
st.sidebar.caption(llm_info['description'])

st.sidebar.markdown("#### 🔍 Retrieval")
top_k = st.sidebar.slider("Top K Results", 1, 5, 3)

compare_llms = st.sidebar.checkbox("Compare All LLMs", value=False)
show_chunk_analysis = st.sidebar.checkbox("Show Chunk Size Analysis", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📄 Upload PDF Documents")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs to knowledge base",
    type=["pdf"],
    accept_multiple_files=True,
    help="PDFs will be added alongside data.txt in the knowledge base"
)

if uploaded_pdfs:
    st.sidebar.success(f"✅ {len(uploaded_pdfs)} PDF(s) loaded")
    for pdf in uploaded_pdfs:
        st.sidebar.caption(f"📄 {pdf.name}")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info:**")
st.sidebar.markdown(f"✅ Embedding: `{embedding_model}`")
st.sidebar.markdown(f"✅ Vector DB: `{backend}`")
st.sidebar.markdown(f"✅ LLM: `{llm_model}`")
st.sidebar.markdown(f"✅ Chunk Size: `{chunk_size}` words")

# ── Load Data ─────────────────────────────────────────────────────────────────
import tempfile, os

def load_knowledge_base(uploaded_pdfs=None):
    file_paths = ["data.txt"]

    # Save uploaded PDFs to temp files and add to file list
    temp_files = []
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf.read())
            tmp.close()
            file_paths.append(tmp.name)
            temp_files.append(tmp.name)

    text = load_documents(file_paths)
    paragraphs = extract_paragraphs(text)

    # Clean up temp files
    for path in temp_files:
        try:
            os.unlink(path)
        except Exception:
            pass

    return paragraphs

# Use uploaded_pdfs as cache key by hashing their names+sizes
pdf_cache_key = tuple((f.name, f.size) for f in uploaded_pdfs) if uploaded_pdfs else ()

@st.cache_data
def cached_load_knowledge_base(pdf_cache_key):
    return load_knowledge_base(uploaded_pdfs)

paragraphs = cached_load_knowledge_base(pdf_cache_key)

# ── Query Input ───────────────────────────────────────────────────────────────
st.markdown("---")
query = st.text_input("💬 Ask your question:", placeholder="e.g. What is deadlock? What is ACID property?")

# ── Chunk Size Analysis Tab ───────────────────────────────────────────────────
if show_chunk_analysis:
    st.markdown("## ✂️ Chunk Size Comparison Analysis")
    comparison = multi_chunk_comparison(paragraphs)

    rows = []
    for size, info in comparison.items():
        rows.append({
            "Chunk Size (words)": size,
            "Total Chunks": info["count"],
            "Avg Words/Chunk": info["avg_words"],
            "Coverage": f"{info['count']} chunks × {info['avg_words']} avg words"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(r["Chunk Size (words)"]) for r in rows],
           [r["Total Chunks"] for r in rows],
           color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
    ax.set_title("Number of Chunks vs Chunk Size")
    ax.set_xlabel("Chunk Size (words)")
    ax.set_ylabel("Number of Chunks")
    st.pyplot(fig)

    st.info("💡 Smaller chunks = more chunks but less context per chunk. Larger chunks = fewer chunks but richer context.")

# ── Main RAG Pipeline ─────────────────────────────────────────────────────────
if st.button("🔍 Ask", type="primary") and query:

    with st.spinner("Processing your query..."):

        # Step 1 — Chunk
        chunks = chunk_text(paragraphs, chunk_size, overlap)

        # Step 2 — Build Index & Search
        if backend == "FAISS":
            index, _ = build_faiss_index(chunks, embedding_model)
            retrieved, scores = search_faiss(index, query, chunks, embedding_model, top_k)
        else:
            collection = build_chroma_collection(chunks, embedding_model)
            retrieved, scores = search_chroma(collection, query, embedding_model, top_k)

        # Step 3 — Build context
        context = "\n\n".join(retrieved)

        # Step 4 — Evaluate
        overlaps = keyword_overlap(query, retrieved)
        metrics = full_evaluation(query, "", context, retrieved, scores)
        confidence = metrics["confidence_level"]

        # Step 5 — Generate Answer
        if confidence == "low":
            answer = "❌ Not found in knowledge base."
        else:
            answer = generate_answer(query, context, llm_model)
            metrics = full_evaluation(query, answer, context, retrieved, scores)

    # ── Output Layout ─────────────────────────────────────────────────────────
    st.markdown("---")

    # Config used
    st.markdown(f"""
    **Configuration Used:**
    `Embedding: {embedding_model}` | `Vector DB: {backend}` | `LLM: {llm_model}` | `Chunk Size: {chunk_size}`
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Retrieved Chunks")
        for i, (c, s, o) in enumerate(zip(retrieved, scores, overlaps)):
            with st.expander(f"Chunk {i+1} | Similarity: {s:.3f} | Keyword Overlap: {o}"):
                st.write(c)

    with col2:
        st.subheader("🤖 Generated Answer")

        if confidence == "low":
            st.error(answer)
        elif confidence == "medium":
            st.warning("⚠️ Partial Match Found")
            st.write(answer)
        else:
            st.success("✅ High Confidence Answer")
            st.write(answer)

    # ── Evaluation Metrics ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Evaluation Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Confidence", metrics["confidence_level"].upper())
    m2.metric("Similarity Score", f"{metrics['similarity_score']:.3f}")
    m3.metric("Faithfulness", f"{metrics['faithfulness']:.3f}")
    m4.metric("Relevancy", f"{metrics['relevancy']:.3f}")
    m5.metric("Keyword Overlap", metrics["keyword_overlap"])

    # Metrics explanation
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        - **Similarity Score**: Semantic distance between query and retrieved chunks (higher = more similar)
        - **Faithfulness**: How much of the answer is supported by the retrieved context (higher = less hallucination)
        - **Relevancy**: How much of the query is covered by retrieved chunks (higher = better retrieval)
        - **Keyword Overlap**: Number of query words found in retrieved chunks
        - **Confidence**: Combined score — Low / Medium / High
        """)

    # ── Similarity Graph ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Similarity Scores Graph")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50" if s > 0.5 else "#FF9800" if s > 0.4 else "#F44336" for s in scores]
    ax.bar([f"Chunk {i+1}" for i in range(len(scores))], scores, color=colors)
    ax.set_title(f"Similarity Scores — {backend} | {embedding_model}")
    ax.set_ylabel("Similarity Score")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='High threshold')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Low threshold')
    ax.legend()
    st.pyplot(fig)

    # ── Compare All LLMs ──────────────────────────────────────────────────────
    if compare_llms and confidence != "low":
        st.markdown("---")
        st.subheader("🧠 LLM Comparison — All Models")
        st.info("Generating answers from all 3 LLMs... this may take a moment.")

        with st.spinner("Running all LLMs..."):
            all_answers = generate_all_models(query, context)

        for model_name, ans in all_answers.items():
            model_info = LLM_MODELS[model_name]
            with st.expander(f"🤖 {model_name} — {model_info['description']}"):
                st.write(ans)

    # ── System Config Summary ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔧 System Configuration Summary")

    config_data = {
        "Component": ["Embedding Model", "Embedding Dimension", "Vector Database", "Chunk Size", "Overlap", "Top K", "LLM Model", "Total Chunks", "PDF Documents"],
        "Value": [
            embedding_model,
            EMBEDDING_MODELS[embedding_model]["dimension"],
            backend,
            chunk_size,
            overlap,
            top_k,
            llm_model,
            len(chunks),
            len(uploaded_pdfs) if uploaded_pdfs else 0
        ]
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True)

# ── Comparative Config Analysis ───────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 System Configurations Comparison Table"):
    st.markdown("### Config 1 vs Config 2")
    comp_data = {
        "Component": ["Embedding Model", "Dimension", "Vector DB", "LLM", "Chunk Size", "Strength"],
        "Config 1": [
            "all-MiniLM-L6-v2",
            "384",
            "FAISS",
            "distilgpt2",
            "500",
            "Fast, lightweight, good baseline"
        ],
        "Config 2": [
            "all-mpnet-base-v2",
            "768",
            "ChromaDB",
            "google/flan-t5-small",
            "200",
            "Better semantics, instruction-tuned LLM"
        ]
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    st.markdown("""
    **Key Differences:**
    - Config 1 uses 384-dim embeddings vs Config 2 uses 768-dim (richer semantic space)
    - Config 1 uses FAISS (faster, in-memory) vs Config 2 uses ChromaDB (persistent, manageable)
    - Config 1 uses distilgpt2 (causal LM) vs Config 2 uses Flan-T5 (instruction-tuned, better answers)
    - Config 1 uses large chunks (500 words) vs Config 2 uses small chunks (200 words)
    """)