"""
Reranker for ordering candidates by relevance.

Uses a default cross-encoder from sentence-transformers or accepts:
- A user-provided CrossEncoder
- A user-provided reranking function with signature: (query: str, candidates: List[str]) -> List[Tuple[str, float]]
"""

from typing import List, Dict, Optional, Tuple, Callable, Protocol
import logging

logger = logging.getLogger(__name__)

# Default reranker model
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerProtocol(Protocol):
    """Protocol for custom reranker functions."""

    def __call__(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: The query text
            candidates: List of candidate texts

        Returns:
            List of (candidate, score) tuples sorted by score descending
        """
        ...


class Reranker:
    """
    Re-ranks candidates by relevance to a query.

    Uses a cross-encoder model by default. Accepts:
    - A sentence-transformers CrossEncoder instance
    - A callable with signature (query, candidates) -> List[(text, score)]

    If nothing is provided, loads the default cross-encoder model.
    """

    def __init__(
        self,
        cross_encoder=None,
        rerank_fn: Optional[Callable[[str, List[str]], List[Tuple[str, float]]]] = None,
        use_default_reranker: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the reranker.

        Args:
            cross_encoder: Optional CrossEncoder from sentence-transformers
            rerank_fn: Optional custom reranking function
            use_default_reranker: If True and no other option provided, load default cross-encoder
            model_name: Model name for default cross-encoder (uses DEFAULT_RERANKER_MODEL if None)
        """
        self.cross_encoder = cross_encoder
        self.rerank_fn = rerank_fn
        self._model_name = model_name or DEFAULT_RERANKER_MODEL

        # Determine which mode to use
        if rerank_fn is not None:
            self._mode = "custom_fn"
            logger.info("Reranker initialized with custom function")
        elif cross_encoder is not None:
            self._mode = "cross_encoder"
            logger.info("Reranker initialized with user-provided cross-encoder")
        elif use_default_reranker:
            self._mode = "cross_encoder"
            self._load_default_cross_encoder()
        else:
            raise ValueError(
                "Must provide one of: cross_encoder, rerank_fn, "
                "or set use_default_reranker=True"
            )

    def _load_default_cross_encoder(self) -> None:
        """Load the default cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder: {self._model_name}")
            self.cross_encoder = CrossEncoder(self._model_name)
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for the default reranker. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self, query: str, candidates: List[str], top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates by relevance to query.

        Args:
            query: The query text
            candidates: List of candidate texts to rank
            top_n: Number of top candidates to return (None = all)

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        if not candidates:
            return []

        if top_n is None:
            top_n = len(candidates)

        if self._mode == "custom_fn":
            results = self.rerank_fn(query, candidates)
            return results[:top_n]
        else:
            return self._rerank_cross_encoder(query, candidates, top_n)

    def _rerank_cross_encoder(
        self, query: str, candidates: List[str], top_n: int
    ) -> List[Tuple[str, float]]:
        """Re-rank using cross-encoder."""
        # Create query-candidate pairs (same format as sentence-transformers)
        pairs = [[query, candidate] for candidate in candidates]

        # Get cross-encoder scores (no progress bar for per-query calls)
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        # Sort by score descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            (candidates[idx], float(score)) for idx, score in indexed_scores[:top_n]
        ]

    def get_top_texts(self, query: str, candidates: List[str], top_n: int) -> List[str]:
        """
        Get only the top-n candidate texts (without scores).

        Args:
            query: Query text
            candidates: Candidate texts
            top_n: Number to return

        Returns:
            List of top-n candidate texts
        """
        results = self.rerank(query, candidates, top_n)
        return [text for text, _ in results]

    def rerank_with_scores(
        self, query: str, candidates: List[str], top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank and return texts WITH scores.

        This is critical for stratified sampling where we need
        to track similarity scores.

        Args:
            query: Query text
            candidates: Candidate texts to rank
            top_n: Number of top candidates to return (None = all)

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        return self.rerank(query, candidates, top_n)

    def rerank_to_bins(
        self,
        query: str,
        candidates: List[str],
        hard_range: Tuple[int, int] = (0, 20),
        medium_range: Tuple[int, int] = (20, 50),
        easy_range: Tuple[int, int] = (50, 200),
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Re-rank and automatically bin into Hard/Medium/Easy.

        Args:
            query: Query text
            candidates: Candidate texts to rank
            hard_range: Index range for hard negatives (top ranks)
            medium_range: Index range for medium negatives
            easy_range: Index range for easy negatives

        Returns:
            Dict with 'hard', 'medium', 'easy' keys, each containing
            a list of (text, score) tuples
        """
        all_ranked = self.rerank_with_scores(query, candidates, top_n=None)

        return {
            "hard": all_ranked[hard_range[0] : hard_range[1]],
            "medium": all_ranked[medium_range[0] : medium_range[1]],
            "easy": all_ranked[easy_range[0] : easy_range[1]],
        }
