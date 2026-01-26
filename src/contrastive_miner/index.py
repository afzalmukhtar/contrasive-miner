"""
Vector Index for fast similarity search.

Supports both FAISS (if available) and NumPy fallback for cosine similarity.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available, falling back to NumPy cosine similarity")


class VectorIndex:
    """
    Fast vector similarity search index.

    Uses FAISS if available, otherwise falls back to NumPy-based
    cosine similarity search.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        use_faiss: bool = True,
    ):
        """
        Initialize the vector index.

        Args:
            embeddings: (N, D) array of embeddings
            metadata: List of metadata dicts, one per embedding
            use_faiss: Whether to use FAISS (if available)
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "must have the same length"
            )

        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata
        self._use_faiss = use_faiss and FAISS_AVAILABLE

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self.normalized_embeddings = self.embeddings / norms

        if self._use_faiss:
            self._build_faiss_index()

        logger.info(
            f"Built VectorIndex with {len(embeddings)} vectors, "
            f"using {'FAISS' if self._use_faiss else 'NumPy'}"
        )

    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast search."""
        dim = self.normalized_embeddings.shape[1]
        # Use inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.normalized_embeddings)

    def search(
        self, query_vec: np.ndarray, top_k: int, exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K similar vectors.

        Args:
            query_vec: (D,) or (1, D) query embedding
            top_k: Number of results to return
            exclude_indices: Set of indices to exclude from results

        Returns:
            List of dicts with 'index', 'score', and metadata fields
        """
        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        exclude_indices = exclude_indices or set()

        # Retrieve more if we need to filter
        retrieve_k = min(top_k + len(exclude_indices) + 10, len(self.embeddings))

        if self._use_faiss:
            scores, indices = self.index.search(query_vec, retrieve_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # NumPy fallback: cosine similarity
            scores = np.dot(self.normalized_embeddings, query_vec.T).flatten()
            indices = np.argsort(-scores)[:retrieve_k]
            scores = scores[indices]

        # Build results, filtering excluded indices
        results = []
        for idx, score in zip(indices, scores):
            if idx == -1:  # FAISS returns -1 for padded results
                continue
            if int(idx) in exclude_indices:
                continue

            result = {"index": int(idx), "score": float(score), **self.metadata[idx]}
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def search_by_text(
        self, query_text: str, model, top_k: int, exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (encodes with provided model).

        Args:
            query_text: Text to search for
            model: Embedding model with .encode() method
            top_k: Number of results
            exclude_indices: Indices to exclude

        Returns:
            List of result dicts
        """
        query_vec = model.encode(query_text, convert_to_numpy=True)
        return self.search(query_vec, top_k, exclude_indices)

    def get_random_indices(
        self,
        n: int,
        exclude_indices: Optional[set] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """
        Get n random indices, excluding specified ones.

        Args:
            n: Number of random indices to get
            exclude_indices: Indices to exclude
            rng: Random number generator (for reproducibility)

        Returns:
            List of random indices
        """
        exclude_indices = exclude_indices or set()
        available = [i for i in range(len(self.embeddings)) if i not in exclude_indices]

        if len(available) == 0:
            return []

        if rng is None:
            rng = np.random.default_rng()

        n = min(n, len(available))
        return list(rng.choice(available, size=n, replace=False))

    def get_metadata_by_index(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific index."""
        return self.metadata[idx]

    def search_with_embeddings(
        self,
        query_vec: np.ndarray,
        top_k: int,
        exclude_indices: Optional[set] = None,
        return_embeddings: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Search and optionally return embeddings for filtering.

        Args:
            query_vec: Query embedding
            top_k: Number of results
            exclude_indices: Indices to exclude
            return_embeddings: Whether to return embeddings

        Returns:
            (results, embeddings) if return_embeddings=True
            (results, None) otherwise
        """
        results = self.search(query_vec, top_k, exclude_indices)

        if return_embeddings and results:
            indices = [r["index"] for r in results]
            result_embeddings = self.normalized_embeddings[indices]
            return results, result_embeddings

        return results, None

    def get_embedding_by_text(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific text.

        Args:
            text: Text to find embedding for

        Returns:
            Normalized embedding or None if not found
        """
        for i, meta in enumerate(self.metadata):
            if meta.get("text") == text:
                return self.normalized_embeddings[i]
        return None

    def batch_get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            Array of embeddings (may be smaller than texts if some not found)
        """
        embeddings = []
        for text in texts:
            emb = self.get_embedding_by_text(text)
            if emb is not None:
                embeddings.append(emb)
        if len(embeddings) == 0:
            return np.array([]).reshape(0, self.normalized_embeddings.shape[1])
        return np.array(embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)
