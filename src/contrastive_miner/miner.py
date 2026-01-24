"""
Semantic Negative Miner - Two-Stage Hard Negative Mining.

The core class that implements:
- Stage 1: Direct document retrieval negatives
- Stage 2: Query similarity negatives

Accepts user-provided embedding model (SentenceTransformer or callable).
Uses default cross-encoder for reranking unless user provides one.
"""

import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Callable, Union
from tqdm import tqdm
import numpy as np

from .models import (
    IntermediateRow,
    TripletRow,
    MinerConfig,
    save_intermediate,
    save_triplets,
    flatten_intermediate_to_triplets,
)
from .index import VectorIndex
from .reranker import Reranker

logger = logging.getLogger(__name__)


class SemanticNegativeMiner:
    """
    Two-stage hard negative mining for contrastive learning.

    Stage 1: Direct Document Retrieval
        - Retrieve chunks for query
        - Filter out positive chunks
        - Re-rank remaining
        - Select hard and random negatives

    Stage 2: Query Similarity
        - Find similar queries
        - Gather their positives
        - Filter out original positives
        - Re-rank and select negatives

    Args:
        embedder: Embedding model. Either:
            - SentenceTransformer instance (with .encode() method)
            - Callable: (texts: Union[str, List[str]]) -> np.ndarray
        config: Mining configuration (optional, uses defaults)
        reranker: Reranker instance. If None, uses default cross-encoder.
        cross_encoder: CrossEncoder for reranking (alternative to reranker)
        rerank_fn: Custom reranking function (alternative to reranker/cross_encoder)
            Signature: (query: str, candidates: List[str]) -> List[Tuple[str, float]]
    """

    def __init__(
        self,
        embedder: Union[Any, Callable],
        config: Optional[MinerConfig] = None,
        reranker: Optional[Reranker] = None,
        cross_encoder: Optional[Any] = None,
        rerank_fn: Optional[Callable[[str, List[str]], List[Tuple[str, float]]]] = None,
    ):
        """
        Initialize the miner.

        Args:
            embedder: Embedding model or function
            config: Mining configuration
            reranker: Optional Reranker instance
            cross_encoder: Optional CrossEncoder for reranking
            rerank_fn: Optional custom reranking function
        """
        self.embedder = embedder
        self.config = config or MinerConfig()

        # Set up reranker
        if reranker is not None:
            self.reranker = reranker
        elif cross_encoder is not None:
            self.reranker = Reranker(cross_encoder=cross_encoder)
        elif rerank_fn is not None:
            self.reranker = Reranker(rerank_fn=rerank_fn)
        else:
            # Use default cross-encoder
            self.reranker = Reranker(use_default_reranker=True)

        # Will be built when data is loaded
        self.chunk_index: Optional[VectorIndex] = None
        self.query_index: Optional[VectorIndex] = None
        self.query_to_positives: Dict[str, Set[str]] = {}
        self.chunk_to_index: Dict[str, int] = {}
        self.all_chunks: List[str] = []

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(42)

    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts using the embedder.

        Supports:
        - SentenceTransformer (with .encode() method)
        - Callable function
        """
        if hasattr(self.embedder, "encode"):
            # SentenceTransformer-like
            return self.embedder.encode(
                texts,
                show_progress_bar=isinstance(texts, list) and len(texts) > 100,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
            )
        elif callable(self.embedder):
            # Custom function
            result = self.embedder(texts)
            return np.array(result) if not isinstance(result, np.ndarray) else result
        else:
            raise ValueError(
                "Embedder must have .encode() method (like SentenceTransformer) "
                "or be callable"
            )

    def build_indices(self, data: List[Dict[str, Any]]) -> None:
        """
        Build chunk and query indices from data.

        Args:
            data: List of dicts with 'anchor'/'query' and 'positives'/'positive'
        """
        logger.info("Building indices from data...")

        # Normalize data format
        normalized_data = self._normalize_data(data)

        # Collect unique chunks and build mappings
        unique_chunks: Set[str] = set()
        queries: List[str] = []
        query_metadata: List[Dict] = []

        for item in normalized_data:
            anchor = item["anchor"]
            positives = item["positives"]

            queries.append(anchor)
            self.query_to_positives[anchor] = set(positives)

            for chunk in positives:
                unique_chunks.add(chunk)

            query_metadata.append({"anchor": anchor, "positives": positives})

        self.all_chunks = list(unique_chunks)
        self.chunk_to_index = {chunk: idx for idx, chunk in enumerate(self.all_chunks)}

        logger.info(
            f"Found {len(self.all_chunks)} unique chunks and {len(queries)} queries"
        )

        # Build chunk index
        logger.info("Encoding chunks...")
        chunk_embeddings = self._encode(self.all_chunks)

        chunk_metadata = [
            {"text": chunk, "chunk_index": idx}
            for idx, chunk in enumerate(self.all_chunks)
        ]

        self.chunk_index = VectorIndex(chunk_embeddings, chunk_metadata)

        # Build query index
        logger.info("Encoding queries...")
        query_embeddings = self._encode(queries)

        self.query_index = VectorIndex(query_embeddings, query_metadata)

        logger.info("Indices built successfully")

    def _normalize_data(self, data: List[Dict]) -> List[Dict]:
        """
        Normalize data to consistent format.

        Supports:
        - {"anchor": "...", "positives": [...]}
        - {"query": "...", "positive": "..."}
        - {"query": "...", "positives": [...]}
        """
        normalized = []
        for item in data:
            anchor = item.get("anchor") or item.get("query")
            positives = item.get("positives")

            if positives is None:
                positive = item.get("positive")
                positives = [positive] if positive else []

            if anchor and positives:
                normalized.append(
                    {
                        "anchor": anchor,
                        "positives": positives
                        if isinstance(positives, list)
                        else [positives],
                    }
                )

        return normalized

    def mine_stage1_doc(
        self, anchor: str, positives: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Stage 1: Mine negatives via direct document retrieval.

        Args:
            anchor: Query text
            positives: Set of positive chunk texts

        Returns:
            (hard_negatives, random_negatives)
        """
        if self.chunk_index is None:
            raise RuntimeError("Call build_indices() first")

        n_retrieve = (
            len(positives)
            + self.config.stage1_n_hard
            + self.config.stage1_retrieve_buffer
        )

        # Get indices of positive chunks to exclude
        exclude_indices = {
            self.chunk_to_index[p] for p in positives if p in self.chunk_to_index
        }

        # Encode query and search
        query_vec = self._encode(anchor)
        results = self.chunk_index.search(query_vec, n_retrieve, exclude_indices)

        # Get candidate chunks (already filtered by exclude_indices)
        candidates = [r["text"] for r in results]

        # Re-rank to get hard negatives
        hard_negatives = []
        if candidates and self.config.stage1_n_hard > 0:
            reranked = self.reranker.get_top_texts(
                anchor, candidates, self.config.stage1_n_hard
            )
            hard_negatives = reranked

        # Sample random negatives (not in positives or hard negatives)
        random_negatives = []
        if self.config.stage1_n_random > 0:
            forbidden = exclude_indices | {
                self.chunk_to_index[h]
                for h in hard_negatives
                if h in self.chunk_to_index
            }
            random_indices = self.chunk_index.get_random_indices(
                self.config.stage1_n_random, forbidden, self.rng
            )
            random_negatives = [self.all_chunks[idx] for idx in random_indices]

        return hard_negatives, random_negatives

    def mine_stage2_query(
        self, anchor: str, positives: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Stage 2: Mine negatives via query similarity.

        Args:
            anchor: Query text
            positives: Set of positive chunk texts

        Returns:
            (hard_negatives, random_negatives)
        """
        if self.query_index is None:
            raise RuntimeError("Call build_indices() first")

        # Encode query and find similar queries
        query_vec = self._encode(anchor)
        similar_queries = self.query_index.search(
            query_vec, self.config.stage2_top_queries + 1
        )

        # Collect positives from similar queries (excluding the anchor itself)
        candidate_chunks: Set[str] = set()
        for result in similar_queries:
            if result["anchor"] == anchor:
                continue  # Skip self

            for chunk in result["positives"]:
                if chunk not in positives:  # Filter out original positives
                    candidate_chunks.add(chunk)

        candidates = list(candidate_chunks)

        # Re-rank to get hard negatives
        hard_negatives = []
        if candidates and self.config.stage2_n_hard > 0:
            reranked = self.reranker.get_top_texts(
                anchor, candidates, self.config.stage2_n_hard
            )
            hard_negatives = reranked

        # Sample random negatives
        random_negatives = []
        if self.config.stage2_n_random > 0:
            # Forbidden: positives + hard negatives + all candidates
            forbidden_chunks = positives | set(hard_negatives) | candidate_chunks
            forbidden_indices = {
                self.chunk_to_index[c]
                for c in forbidden_chunks
                if c in self.chunk_to_index
            }

            random_indices = self.chunk_index.get_random_indices(
                self.config.stage2_n_random, forbidden_indices, self.rng
            )
            random_negatives = [self.all_chunks[idx] for idx in random_indices]

        return hard_negatives, random_negatives

    def mine_row(self, anchor: str, positives: List[str]) -> List[IntermediateRow]:
        """
        Mine negatives for a single anchor and its positives.

        Creates one IntermediateRow per positive (exploded).

        Args:
            anchor: Query text
            positives: List of positive chunks

        Returns:
            List of IntermediateRows (one per positive)
        """
        positives_set = set(positives)

        # Stage 1: Document retrieval negatives
        hard_doc, random_doc = self.mine_stage1_doc(anchor, positives_set)

        # Stage 2: Query similarity negatives
        hard_query, random_query = self.mine_stage2_query(anchor, positives_set)

        # Explode: one row per positive
        rows = []
        for positive in positives:
            rows.append(
                IntermediateRow(
                    anchor=anchor,
                    positive=positive,
                    hard_neg_doc=hard_doc.copy(),
                    random_neg_doc=random_doc.copy(),
                    hard_neg_query=hard_query.copy(),
                    random_neg_query=random_query.copy(),
                )
            )

        return rows

    def mine_dataset(
        self,
        data: List[Dict[str, Any]],
        intermediate_path: Optional[str] = None,
        final_path: Optional[str] = None,
        build_indices: bool = True,
    ) -> Tuple[List[IntermediateRow], List[TripletRow]]:
        """
        Mine negatives for entire dataset.

        Args:
            data: Input data with anchors and positives
            intermediate_path: Path to save intermediate results (stage-specific)
            final_path: Path to save final triplets
            build_indices: Whether to rebuild indices (False if already built)

        Returns:
            (intermediate_rows, triplet_rows)
        """
        if build_indices:
            self.build_indices(data)

        normalized_data = self._normalize_data(data)

        intermediate_rows: List[IntermediateRow] = []

        logger.info("Mining negatives...")
        for item in tqdm(normalized_data, desc="Mining"):
            rows = self.mine_row(item["anchor"], item["positives"])
            intermediate_rows.extend(rows)

        # Save intermediate
        if intermediate_path:
            logger.info(f"Saving intermediate results to {intermediate_path}")
            save_intermediate(intermediate_rows, intermediate_path)

        # Flatten to final format
        triplet_rows = flatten_intermediate_to_triplets(intermediate_rows)

        # Save final
        if final_path:
            logger.info(f"Saving final triplets to {final_path}")
            save_triplets(triplet_rows, final_path)

        logger.info(
            f"Mining complete: {len(intermediate_rows)} intermediate rows, "
            f"{len(triplet_rows)} triplet rows"
        )

        return intermediate_rows, triplet_rows

    @classmethod
    def from_file(
        cls,
        data_path: str,
        embedder: Union[Any, Callable],
        config: Optional[MinerConfig] = None,
        **kwargs,
    ) -> "SemanticNegativeMiner":
        """
        Create miner and load data from file.

        Args:
            data_path: Path to JSON/JSONL file with data
            embedder: Embedding model or function
            config: Mining configuration
            **kwargs: Additional arguments (cross_encoder, rerank_fn, etc.)

        Returns:
            Initialized miner with built indices
        """
        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        miner = cls(embedder, config, **kwargs)
        miner.build_indices(data)

        return miner
