"""
Similarity utilities for hard negative mining.

Provides functions for computing cosine similarity and filtering candidates.
"""

import numpy as np
from typing import List, Tuple, Union

# Try to import torch for tensor support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def cosine_similarity(
    vec1: Union[np.ndarray, "torch.Tensor"],
    vec2: Union[np.ndarray, "torch.Tensor"],
) -> float:
    """
    Compute cosine similarity between two vectors.

    Handles both NumPy and PyTorch tensors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if TORCH_AVAILABLE and isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().numpy()
    if TORCH_AVAILABLE and isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().numpy()

    # Flatten if needed
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    # Compute
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def batch_cosine_similarity(
    query_vec: Union[np.ndarray, "torch.Tensor"],
    candidate_vecs: Union[np.ndarray, "torch.Tensor"],
) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple candidates.

    Args:
        query_vec: Shape (D,) or (1, D)
        candidate_vecs: Shape (N, D)

    Returns:
        Similarities: Shape (N,)
    """
    if TORCH_AVAILABLE and isinstance(query_vec, torch.Tensor):
        query_vec = query_vec.cpu().numpy()
    if TORCH_AVAILABLE and isinstance(candidate_vecs, torch.Tensor):
        candidate_vecs = candidate_vecs.cpu().numpy()

    query_vec = query_vec.flatten()

    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    candidate_norms = candidate_vecs / (
        np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8
    )

    # Compute
    similarities = np.dot(candidate_norms, query_norm)

    return similarities


def filter_by_similarity(
    query_embedding: np.ndarray,
    positive_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_texts: List[str],
    positive_text: str,
    threshold: float = 0.75,
    query_threshold: float = 0.9,
) -> Tuple[List[str], np.ndarray, int, np.ndarray]:
    """
    Filter candidates that are too similar to positive or query.

    Args:
        query_embedding: Embedding of the query
        positive_embedding: Embedding of the positive
        candidate_embeddings: Embeddings of candidates (N, D)
        candidate_texts: Text of each candidate
        positive_text: Text of the positive (for exact match check)
        threshold: Similarity threshold for positive (default 0.75)
        query_threshold: Similarity threshold for query (default 0.9)

    Returns:
        Tuple of:
        - valid_texts: List of valid candidate texts
        - valid_embeddings: Embeddings of valid candidates
        - filtered_count: Number of filtered candidates
        - query_similarities: Similarity scores to query for valid candidates
    """
    n_candidates = len(candidate_texts)

    if n_candidates == 0:
        return [], np.array([]).reshape(0, positive_embedding.shape[0]), 0, np.array([])

    # Compute similarities
    pos_similarities = batch_cosine_similarity(positive_embedding, candidate_embeddings)
    query_similarities = batch_cosine_similarity(query_embedding, candidate_embeddings)

    # Filter
    valid_indices = []
    filtered_count = 0

    for i in range(n_candidates):
        # Skip exact match
        if candidate_texts[i] == positive_text:
            filtered_count += 1
            continue

        # Skip if too similar to positive
        if pos_similarities[i] > threshold:
            filtered_count += 1
            continue

        # Skip if too similar to query (very strict)
        if query_similarities[i] > query_threshold:
            filtered_count += 1
            continue

        valid_indices.append(i)

    if len(valid_indices) == 0:
        return (
            [],
            np.array([]).reshape(0, positive_embedding.shape[0]),
            filtered_count,
            np.array([]),
        )

    valid_texts = [candidate_texts[i] for i in valid_indices]
    valid_embeddings = candidate_embeddings[valid_indices]
    valid_query_similarities = query_similarities[valid_indices]

    return valid_texts, valid_embeddings, filtered_count, valid_query_similarities
