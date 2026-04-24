"""
embedding.py
============
Embedding generation module — supports multiple embedding models.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Available embedding models — all CPU friendly
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast, lightweight — good for general QA"
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "dimension": 768,
        "description": "Higher dimension — better semantic understanding"
    },
    "paraphrase-MiniLM-L3-v2": {
        "name": "paraphrase-MiniLM-L3-v2",
        "dimension": 384,
        "description": "Smallest and fastest — good for quick retrieval"
    }
}

# Cache loaded models
_model_cache = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache an embedding model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def generate_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed
        model_name: Name of the sentence-transformer model

    Returns:
        numpy array of embeddings
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string."""
    model = get_embedding_model(model_name)
    return model.encode([query])[0].astype(np.float32)


def get_model_info(model_name: str) -> dict:
    """Return metadata about an embedding model."""
    return EMBEDDING_MODELS.get(model_name, {
        "name": model_name,
        "dimension": "unknown",
        "description": "Custom model"
    })
