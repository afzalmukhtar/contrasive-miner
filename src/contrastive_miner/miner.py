"""
Semantic Negative Miner - Two-Stage Hard Negative Mining.

The core class that implements:
- Stage 1: Direct document retrieval negatives with stratified sampling
- Stage 2: Topic neighbor mining via query similarity

Accepts user-provided embedding model (SentenceTransformer or callable).
Uses default cross-encoder for reranking unless user provides one.
"""

import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Callable, Union
from tqdm import tqdm
import numpy as np

from .models import (
    TripletRow,
    MinerConfig,
    save_triplets,
    CandidateNegative,
    MiningStats,
)
from .index import VectorIndex
from .reranker import Reranker
from .utils.similarity import filter_by_similarity

logger = logging.getLogger(__name__)


class SemanticNegativeMiner:
    """
    Two-stage hard negative mining for contrastive learning with stratified sampling.

    Stage 1: Direct Document Retrieval
        - Retrieve chunks for query
        - Filter out positive chunks
        - Re-rank remaining
        - Stratified sampling into Hard/Medium/Easy buckets

    Stage 2: Topic Neighbor Mining
        - Find similar queries
        - Gather their positives
        - Filter out original positives and high similarity chunks
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
            # Use default cross-encoder with model from config
            self.reranker = Reranker(
                use_default_reranker=True,
                model_name=self.config.cross_encoder_model,
            )

        # Will be built when data is loaded
        self.chunk_index: Optional[VectorIndex] = None
        self.query_index: Optional[VectorIndex] = None
        self.query_to_positives: Dict[str, Set[str]] = {}
        self.chunk_to_index: Dict[str, int] = {}
        self.all_chunks: List[str] = []

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(self.config.random_seed)

        # Internal flag for progress bars
        self._suppress_progress = False

    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using the embedder."""
        if hasattr(self.embedder, "encode"):
            return self.embedder.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=not self._suppress_progress
                and self.config.show_progress,
                device="cuda" if self.config.use_gpu else "cpu",
            )
        else:
            return self.embedder(texts)

    def _sample_true_random_negatives(
        self,
        positives: Set[str],
        positive_embeddings: np.ndarray,
        n_samples: int,
        similarity_threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Sample true random negatives from the corpus with low similarity to ALL positives.

        These are completely unrelated chunks, useful for TripletLoss without
        in-batch negatives. For MNRL, the semi-hard negatives (ranks 50-200) are
        typically better as in-batch negatives provide the "true random" signal.

        Args:
            positives: Set of positive texts to exclude
            positive_embeddings: Embeddings of ALL positives (M, D) to measure similarity
            n_samples: Number of negatives to sample
            similarity_threshold: Maximum similarity to ANY positive (lower = more dissimilar)

        Returns:
            List of (text, max_similarity_to_any_positive) tuples
        """
        if self.chunk_index is None or len(self.all_chunks) == 0:
            return []

        # Ensure positive_embeddings is 2D
        if positive_embeddings.ndim == 1:
            positive_embeddings = positive_embeddings.reshape(1, -1)

        # Sample more candidates than needed to account for filtering
        sample_size = min(n_samples * 10, len(self.all_chunks))
        candidate_indices = self.rng.choice(
            len(self.all_chunks), size=sample_size, replace=False
        )

        # Filter out positives and compute similarity
        valid_candidates = []
        for idx in candidate_indices:
            text = self.all_chunks[idx]
            if text in positives:
                continue

            # Get embedding and compute max similarity to ANY positive
            chunk_embedding = self.chunk_index.normalized_embeddings[idx]
            # Both are normalized, so dot product = cosine similarity
            max_similarity = 0.0
            for pos_emb in positive_embeddings:
                sim = float(np.dot(pos_emb.flatten(), chunk_embedding.flatten()))
                max_similarity = max(max_similarity, sim)

            # Keep only chunks with low similarity to ALL positives
            if max_similarity < similarity_threshold:
                valid_candidates.append((text, max_similarity))

            if len(valid_candidates) >= n_samples:
                break

        return valid_candidates

    def build_indices(self, data: List[Dict[str, Any]]) -> None:
        """
        Build chunk and query indices from data.

        Args:
            data: List of dicts with 'anchor'/'query' and 'positives'/'positive'
        """
        logger.info("Building indices...")

        # Collect all unique chunks and queries
        chunks = set()
        queries = []

        # Pre-scan to normalize keys
        normalized = self._normalize_data(data)

        for item in tqdm(normalized, desc="Processing data"):
            anchor = item["anchor"]
            positives = item["positives"]

            queries.append(anchor)
            for p in positives:
                chunks.add(p)

            # Map query to positives for exclusion
            if anchor not in self.query_to_positives:
                self.query_to_positives[anchor] = set()
            self.query_to_positives[anchor].update(positives)

        self.all_chunks = list(chunks)
        self.chunk_to_index = {chunk: i for i, chunk in enumerate(self.all_chunks)}

        logger.info(f"Encoding {len(self.all_chunks)} unique chunks...")
        chunk_embeddings = self._encode(self.all_chunks)

        logger.info(f"Encoding {len(queries)} queries...")
        query_vecs = self._encode(queries)

        # Build FAISS indices
        self.chunk_index = VectorIndex(
            embeddings=chunk_embeddings,
            metadata=[{"text": t} for t in self.all_chunks],
            use_faiss=self.config.use_gpu,  # Map use_gpu to use_faiss
        )

        # Store minimal metadata for query index (just anchor and positives reference)
        # to avoid duplicating all text data
        query_metadata = [
            {"anchor": item["anchor"], "positives": item["positives"]}
            for item in normalized
        ]
        self.query_index = VectorIndex(
            embeddings=query_vecs,
            metadata=query_metadata,
            use_faiss=self.config.use_gpu,
        )
        logger.info("Indices built successfully.")

    def _normalize_data(self, data: List[Dict], warn: bool = True) -> List[Dict[str, Any]]:
        """Normalize data to consistent format."""
        normalized = []
        skipped = 0
        
        for item in data:
            anchor = item.get("anchor") or item.get("query")
            positives = item.get("positives") or item.get("positive")

            if isinstance(positives, str):
                positives = [positives]

            if anchor and positives:
                normalized.append(
                    {"anchor": anchor, "positives": positives}  # Keep as list
                )
            else:
                skipped += 1
        
        if warn and skipped > 0:
            logger.warning(
                f"Skipped {skipped} items with missing anchor or positives. "
                f"Expected format: {{'anchor': str, 'positives': list}} or "
                f"{{'query': str, 'positive': str}}"
            )

        return normalized

    def mine_stage1_stratified(
        self, anchor: str, positives: Set[str]
    ) -> Tuple[List[CandidateNegative], MiningStats]:
        """
        Stage 1: Stratified Hard Negative Mining with Validation.

        1. Retrieve Top K from FAISS
        2. Filter False Positives (>threshold similarity)
        3. Cross-Encoder Rerank
        4. Stratified Sampling into Hard/Medium/Easy buckets

        Args:
            anchor: Query text
            positives: Set of positive chunk texts

        Returns:
            (candidates, stats)
        """
        stats = MiningStats()

        if self.chunk_index is None:
            raise RuntimeError("Call build_indices() first")

        # Step 1: Retrieve Top K from FAISS
        query_vec = self._encode(anchor)
        results, result_embeddings = self.chunk_index.search_with_embeddings(
            query_vec,
            self.config.stage1_retrieve_k,
            exclude_indices=None,
            return_embeddings=True,
        )

        stats.stage1_retrieved = len(results)

        if not results or result_embeddings is None:
            # logger.warning(f"No results for query: {anchor[:50]}")
            return [], stats

        # Step 2: Filter False Positives
        candidate_texts = [r["text"] for r in results]

        # Get embeddings for ALL positives for similarity check
        positive_texts = list(positives)
        positive_embeddings = []
        for pos_text in positive_texts:
            pos_emb = self.chunk_index.get_embedding_by_text(pos_text)
            if pos_emb is None:
                pos_emb = self._encode(pos_text)
            positive_embeddings.append(pos_emb)
        positive_embeddings = np.array(positive_embeddings)

        # Filter using similarity against ALL positives
        valid_texts, valid_embeddings, filtered_count, query_similarities = (
            filter_by_similarity(
                query_vec,
                positive_embeddings,
                result_embeddings,
                candidate_texts,
                positive_texts,
                threshold=self.config.stage1_similarity_threshold,
            )
        )

        stats.stage1_filtered_similar = filtered_count
        stats.stage1_after_filter = len(valid_texts)

        if len(valid_texts) == 0:
            return [], stats

        # Step 3: Cross-Encoder Rerank
        ranked = self.reranker.rerank_to_bins(
            anchor,
            valid_texts,
            hard_range=self.config.stage1_hard_range,
            medium_range=self.config.stage1_medium_range,
            easy_range=self.config.stage1_easy_range,
        )

        # Step 4: Stratified Sampling
        candidates = []
        cfg = self.config

        # Sample Hard
        hard_pool = ranked["hard"]
        n_hard = min(len(hard_pool), cfg.hard_multiplier * cfg.multiplier)
        if n_hard > 0 and len(hard_pool) > 0:
            hard_indices = self.rng.choice(
                len(hard_pool), size=min(n_hard, len(hard_pool)), replace=False
            )
            for idx in hard_indices:
                text, score = hard_pool[idx]
                candidates.append(
                    CandidateNegative(
                        text=text,
                        score=score,
                        source="stage1_hard",
                        query_similarity=score,
                    )
                )
            stats.stage1_hard_sampled = len(hard_indices)
            hard_sims = [hard_pool[i][1] for i in hard_indices]
            stats.avg_hard_similarity = float(np.mean(hard_sims))

        # Sample Medium
        medium_pool = ranked["medium"]
        n_medium = min(len(medium_pool), cfg.medium_multiplier * cfg.multiplier)
        if n_medium > 0 and len(medium_pool) > 0:
            medium_indices = self.rng.choice(
                len(medium_pool), size=min(n_medium, len(medium_pool)), replace=False
            )
            for idx in medium_indices:
                text, score = medium_pool[idx]
                candidates.append(
                    CandidateNegative(
                        text=text,
                        score=score,
                        source="stage1_medium",
                        query_similarity=score,
                    )
                )
            stats.stage1_medium_sampled = len(medium_indices)
            medium_sims = [medium_pool[i][1] for i in medium_indices]
            stats.avg_medium_similarity = float(np.mean(medium_sims))

        # Sample Easy (or True Random if configured)
        n_easy = cfg.easy_multiplier * cfg.multiplier

        if cfg.use_true_random_easy:
            # True Random: Sample from entire corpus with low similarity to ALL positives
            true_random_candidates = self._sample_true_random_negatives(
                positives=positives,
                positive_embeddings=positive_embeddings,
                n_samples=n_easy,
                similarity_threshold=cfg.true_random_similarity_threshold,
            )
            for text, similarity in true_random_candidates:
                candidates.append(
                    CandidateNegative(
                        text=text,
                        score=similarity,
                        source="stage1_true_random",
                        query_similarity=similarity,
                    )
                )
            stats.stage1_easy_sampled = len(true_random_candidates)
            if true_random_candidates:
                stats.avg_easy_similarity = float(
                    np.mean([sim for _, sim in true_random_candidates])
                )
        else:
            # Semi-Hard: Use rank 50-200 pool (default for MNRL)
            easy_pool = ranked["easy"]
            if n_easy > 0 and len(easy_pool) > 0:
                easy_indices = self.rng.choice(
                    len(easy_pool), size=min(n_easy, len(easy_pool)), replace=False
                )
                for idx in easy_indices:
                    text, score = easy_pool[idx]
                    candidates.append(
                        CandidateNegative(
                            text=text,
                            score=score,
                            source="stage1_easy",
                            query_similarity=score,
                        )
                    )
                stats.stage1_easy_sampled = len(easy_indices)
                easy_sims = [easy_pool[i][1] for i in easy_indices]
                stats.avg_easy_similarity = float(np.mean(easy_sims))

        return candidates, stats

    def mine_stage2_topic_neighbors(
        self, anchor: str, positives: Set[str]
    ) -> Tuple[List[CandidateNegative], MiningStats]:
        """
        Stage 2: Topic Neighbor Mining.

        1. Find similar queries
        2. Extract their positives
        3. Filter high similarity to original positive
        4. Cross-encoder rerank
        5. Random sample

        Args:
            anchor: Query text
            positives: Set of positive chunk texts

        Returns:
            (candidates, stats)
        """
        stats = MiningStats()

        if self.query_index is None:
            raise RuntimeError("Call build_indices() first")

        # Step 1: Find Similar Queries
        query_vec = self._encode(anchor)
        similar_queries = self.query_index.search(
            query_vec, self.config.stage2_top_queries + 1
        )

        # Step 2: Collect Their Positives
        candidate_chunks: Set[str] = set()
        for result in similar_queries:
            if result["anchor"] == anchor:
                continue  # Skip self

            for chunk in result["positives"]:
                # Exclude original positives
                if chunk not in positives:
                    candidate_chunks.add(chunk)

        candidate_list = list(candidate_chunks)
        stats.stage2_candidates = len(candidate_list)

        if len(candidate_list) == 0:
            return [], stats

        # Step 3: Filter High Similarity to ALL Original Positives
        positive_texts = list(positives)
        positive_embeddings = []
        for pos_text in positive_texts:
            pos_emb = self.chunk_index.get_embedding_by_text(pos_text)
            if pos_emb is None:
                pos_emb = self._encode(pos_text)
            positive_embeddings.append(pos_emb)
        positive_embeddings = np.array(positive_embeddings)

        # Get embeddings for candidates (with mask to track which were found)
        candidate_embeddings, found_mask = self.chunk_index.batch_get_embeddings(
            candidate_list, return_mask=True
        )

        if len(candidate_embeddings) == 0:
            logger.debug("Stage 2: Could not find embeddings for candidates")
            return [], stats
        
        # Filter candidate_list to only include found texts
        if found_mask is not None:
            candidate_list = [t for t, found in zip(candidate_list, found_mask) if found]

        # Filter against ALL positives
        valid_texts, valid_embeddings, filtered_count, query_similarities = (
            filter_by_similarity(
                query_vec,
                positive_embeddings,
                candidate_embeddings,
                candidate_list,
                positive_texts,
                threshold=self.config.stage2_similarity_threshold,
            )
        )

        stats.stage2_filtered_similar = filtered_count

        if len(valid_texts) == 0:
            return [], stats

        # Step 4: Cross-Encoder Rerank
        ranked = self.reranker.rerank_with_scores(anchor, valid_texts, top_n=None)

        # Step 5: Random Sample
        cfg = self.config
        n_sample = min(len(ranked), cfg.stage2_multiplier * cfg.multiplier)

        candidates = []
        if n_sample > 0:
            sample_indices = self.rng.choice(len(ranked), size=n_sample, replace=False)
            for idx in sample_indices:
                text, score = ranked[idx]
                candidates.append(
                    CandidateNegative(
                        text=text,
                        score=score,
                        source="stage2_topic_neighbor",
                        query_similarity=score,
                    )
                )
            stats.stage2_sampled = n_sample

            stage2_sims = [ranked[i][1] for i in sample_indices]
            stats.avg_stage2_similarity = float(np.mean(stage2_sims))

        return candidates, stats

    def mine_row(
        self, anchor: str, positives: List[str]
    ) -> Tuple[List[TripletRow], MiningStats]:
        """
        Mine negatives for a single anchor using stratified sampling.

        This is the main orchestrator that combines Stage 1 and Stage 2.

        Args:
            anchor: Query text
            positives: List of positive chunks

        Returns:
            (triplet_rows, combined_stats)
        """
        positives_set = set(positives)
        combined_stats = MiningStats()

        # Stage 1: Stratified Mining
        stage1_candidates, stage1_stats = self.mine_stage1_stratified(
            anchor, positives_set
        )

        # Stage 2: Topic Neighbors
        stage2_candidates, stage2_stats = self.mine_stage2_topic_neighbors(
            anchor, positives_set
        )

        # Combine candidates
        all_candidates = stage1_candidates + stage2_candidates

        # Merge stats
        combined_stats.stage1_retrieved = stage1_stats.stage1_retrieved
        combined_stats.stage1_filtered_similar = stage1_stats.stage1_filtered_similar
        combined_stats.stage1_after_filter = stage1_stats.stage1_after_filter
        combined_stats.stage1_hard_sampled = stage1_stats.stage1_hard_sampled
        combined_stats.stage1_medium_sampled = stage1_stats.stage1_medium_sampled
        combined_stats.stage1_easy_sampled = stage1_stats.stage1_easy_sampled
        combined_stats.stage2_candidates = stage2_stats.stage2_candidates
        combined_stats.stage2_filtered_similar = stage2_stats.stage2_filtered_similar
        combined_stats.stage2_sampled = stage2_stats.stage2_sampled
        combined_stats.avg_hard_similarity = stage1_stats.avg_hard_similarity
        combined_stats.avg_medium_similarity = stage1_stats.avg_medium_similarity
        combined_stats.avg_easy_similarity = stage1_stats.avg_easy_similarity
        combined_stats.avg_stage2_similarity = stage2_stats.avg_stage2_similarity

        combined_stats.total_before_dedup = len(all_candidates)

        # Deduplicate
        seen_texts = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand.text not in seen_texts and cand.text not in positives_set:
                seen_texts.add(cand.text)
                unique_candidates.append(cand)

        combined_stats.total_after_dedup = len(unique_candidates)

        # Create rows (one per positive)
        rows = []
        for positive in positives:
            negative_texts = [c.text for c in unique_candidates]
            negative_sources = [c.source for c in unique_candidates]
            negative_similarities = [c.score for c in unique_candidates]

            rows.append(
                TripletRow(
                    anchor=anchor,
                    positive=positive,
                    negatives=negative_texts,
                    negative_sources=negative_sources,
                    negative_similarities=negative_similarities,
                    mining_stats=combined_stats,
                )
            )

        return rows, combined_stats

    def mine_dataset(
        self,
        data: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        build_indices: bool = True,
        log_stats: bool = True,
    ) -> List[TripletRow]:
        """
        Mine negatives for entire dataset using stratified mining.

        Args:
            data: Input data with anchors and positives
            output_path: Path to save triplets (JSON lines)
            build_indices: Whether to rebuild indices
            log_stats: Whether to log aggregate statistics

        Returns:
            List of TripletRow
        """
        if build_indices:
            self.build_indices(data)

        normalized_data = self._normalize_data(data)

        all_rows: List[TripletRow] = []
        aggregate_stats = MiningStats()

        logger.info("Mining negatives with stratified sampling...")
        self._suppress_progress = True
        try:
            for item in tqdm(normalized_data, desc="Mining"):
                rows, stats = self.mine_row(item["anchor"], item["positives"])
                all_rows.extend(rows)

                # Aggregate stats (counts)
                aggregate_stats.stage1_retrieved += stats.stage1_retrieved
                aggregate_stats.stage1_filtered_similar += stats.stage1_filtered_similar
                aggregate_stats.stage1_after_filter += stats.stage1_after_filter
                aggregate_stats.stage1_hard_sampled += stats.stage1_hard_sampled
                aggregate_stats.stage1_medium_sampled += stats.stage1_medium_sampled
                aggregate_stats.stage1_easy_sampled += stats.stage1_easy_sampled
                aggregate_stats.stage2_candidates += stats.stage2_candidates
                aggregate_stats.stage2_filtered_similar += stats.stage2_filtered_similar
                aggregate_stats.stage2_sampled += stats.stage2_sampled
                aggregate_stats.total_before_dedup += stats.total_before_dedup
                aggregate_stats.total_after_dedup += stats.total_after_dedup
                
                # Accumulate similarity sums for averaging later
                aggregate_stats.avg_hard_similarity += stats.avg_hard_similarity
                aggregate_stats.avg_medium_similarity += stats.avg_medium_similarity
                aggregate_stats.avg_easy_similarity += stats.avg_easy_similarity
                aggregate_stats.avg_stage2_similarity += stats.avg_stage2_similarity
        finally:
            self._suppress_progress = False

        # Compute averages for quality metrics (divide accumulated sums)
        n = len(normalized_data)
        if n > 0:
            aggregate_stats.avg_hard_similarity /= n
            aggregate_stats.avg_medium_similarity /= n
            aggregate_stats.avg_easy_similarity /= n
            aggregate_stats.avg_stage2_similarity /= n

        if log_stats:
            aggregate_stats.log_summary(logger)

        # Save if path provided
        if output_path:
            logger.info(f"Saving triplets to {output_path}")
            save_triplets(all_rows, output_path)

        logger.info(f"Mining complete: {len(all_rows)} triplet rows")

        return all_rows

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
