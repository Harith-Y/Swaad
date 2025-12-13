"""
Embedding and vector operations using SentenceTransformer.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import SENTENCE_TRANSFORMER_MODEL
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Global model instance
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _embedding_model


def embed_text(text: str) -> List[float]:
    """Generate embedding vector for text."""
    model = get_embedding_model()
    return model.encode(text).tolist()


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def combine_vectors(primary: List[float], secondary: List[float], secondary_weight: float = 0.35) -> List[float]:
    """Combine two vectors with weighted average."""
    if not primary:
        return secondary or [0.0] * 6
    if not secondary:
        return primary
    
    primary_weight = 1.0 - secondary_weight
    combined = []
    for p, s in zip(primary, secondary):
        combined.append(p * primary_weight + s * secondary_weight)
    return combined


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    if not vec:
        return vec
    
    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    
    return [x / magnitude for x in vec]

