"""
Data models for Contrastive Miner.

Defines the data structures for:
- IntermediateRow: Stores stage-specific negatives
- TripletRow: Final flattened format for training
- MinerConfig: Configuration for the mining process
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


@dataclass
class TripletRow:
    """
    Final flattened format for Triplet Loss training.

    Each row has one positive and multiple negatives that can be
    used with MultipleNegativesRankingLoss or TripletLoss.

    Optional metadata fields support quality analysis from stratified mining.
    """

    anchor: str
    positive: str
    negatives: List[str] = field(default_factory=list)

    # Optional metadata for quality analysis (from stratified mining)
    negative_sources: List[str] = field(default_factory=list)
    negative_similarities: List[float] = field(default_factory=list)
    mining_stats: Optional["MiningStats"] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "anchor": self.anchor,
            "positive": self.positive,
            "negatives": self.negatives,
        }
        # Only include metadata if present
        if self.negative_sources:
            result["negative_sources"] = self.negative_sources
        if self.negative_similarities:
            result["negative_similarities"] = self.negative_similarities
        if self.mining_stats:
            result["mining_stats"] = asdict(self.mining_stats)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TripletRow":
        """Create from dictionary."""
        stats = None
        if "mining_stats" in data and data["mining_stats"]:
            stats = MiningStats(**data["mining_stats"])
        return cls(
            anchor=data["anchor"],
            positive=data["positive"],
            negatives=data.get("negatives", []),
            negative_sources=data.get("negative_sources", []),
            negative_similarities=data.get("negative_similarities", []),
            mining_stats=stats,
        )


class MinerConfig(BaseModel):
    """
    Configuration for the SemanticNegativeMiner with stratified sampling.

    Stage 1: Direct document retrieval with Hard/Medium/Easy bucketing
    Stage 2: Topic neighbor mining from similar queries
    """

    # Stage 1: Direct Retrieval with stratified sampling
    stage1_retrieve_k: int = Field(
        default=200,
        description="Number of candidates to retrieve for stratified mining",
    )
    stage1_similarity_threshold: float = Field(
        default=0.6,
        description="Filter candidates with similarity > threshold to positive",
    )
    stage1_hard_range: Tuple[int, int] = Field(
        default=(0, 20), description="Index range for hard negatives (top ranks)"
    )
    stage1_medium_range: Tuple[int, int] = Field(
        default=(20, 50), description="Index range for medium negatives"
    )
    stage1_easy_range: Tuple[int, int] = Field(
        default=(50, 200), description="Index range for easy negatives"
    )
    use_true_random_easy: bool = Field(
        default=False,
        description="If True, sample 'easy' negatives randomly from the entire corpus "
        "(with low similarity to positives) instead of ranks 50-200. "
        "Use this for TripletLoss without in-batch negatives. "
        "Keep False for MNRL which benefits from semi-hard negatives.",
    )
    true_random_similarity_threshold: float = Field(
        default=0.3,
        description="Max similarity to positive for true random negatives. "
        "Lower values ensure more dissimilar (easier) negatives.",
    )

    # Stage 2: Topic Neighbors
    stage2_top_queries: int = Field(
        default=10, description="Number of similar queries to retrieve"
    )
    stage2_similarity_threshold: float = Field(
        default=0.6, description="Filter threshold for stage 2 candidates"
    )

    # Sampling multipliers
    multiplier: int = Field(default=1, description="Base multiplier for each bin")
    hard_multiplier: int = Field(default=2, description="Multiplier for hard negatives")
    medium_multiplier: int = Field(
        default=2, description="Multiplier for medium negatives"
    )
    easy_multiplier: int = Field(default=2, description="Multiplier for easy negatives")
    stage2_multiplier: int = Field(default=2, description="Multiplier for stage 2")

    # Cross-Encoder
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Cross-encoder model for reranking",
    )
    cross_encoder_batch_size: int = Field(
        default=32, description="Batch size for cross-encoder"
    )

    # Embedding
    batch_size: int = Field(
        default=32, description="Batch size for embedding computation"
    )

    # Performance
    use_faiss: bool = Field(
        default=True, description="Whether to use FAISS for vector search (faster)"
    )
    embedding_device: str = Field(
        default="cuda",
        description="Device for embedding model ('cuda', 'cpu', or 'mps')",
    )
    show_progress: bool = Field(default=True, description="Show progress bars")
    skip_empty_results: bool = Field(
        default=False,
        description="If True, skip rows with no negatives. If False, use fallback random sampling.",
    )
    fallback_negative_count: int = Field(
        default=5,
        description="Number of random negatives to sample as fallback when mining finds none",
    )

    # Reproducibility
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()


def save_triplets(rows: List[TripletRow], path: str) -> None:
    """Save triplet rows to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def load_triplets(path: str) -> List[TripletRow]:
    """Load triplet rows from JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(TripletRow.from_dict(json.loads(line)))
    return rows


# =============================================================================
# Advanced Mining Data Models
# =============================================================================


@dataclass
class CandidateNegative:
    """
    Represents a candidate negative with metadata.

    Tracks the source and similarity scores for analysis.
    """

    text: str
    score: float  # From retriever or cross-encoder
    source: str  # 'stage1_hard', 'stage1_medium', 'stage1_easy', 'stage2'
    query_similarity: float = 0.0
    positive_similarity: float = 0.0


@dataclass
class MiningStats:
    """
    Track mining statistics for debugging and quality analysis.

    Captures counts and average similarities at each stage.
    """

    # Stage 1
    stage1_retrieved: int = 0
    stage1_filtered_exact: int = 0
    stage1_filtered_similar: int = 0
    stage1_after_filter: int = 0
    stage1_hard_sampled: int = 0
    stage1_medium_sampled: int = 0
    stage1_easy_sampled: int = 0

    # Stage 2
    stage2_candidates: int = 0
    stage2_filtered_similar: int = 0
    stage2_sampled: int = 0

    # Final
    total_before_dedup: int = 0
    total_after_dedup: int = 0
    fallback_sampled: int = 0  # Random fallback when mining finds nothing

    # Quality metrics
    avg_hard_similarity: float = 0.0
    avg_medium_similarity: float = 0.0
    avg_easy_similarity: float = 0.0
    avg_stage2_similarity: float = 0.0

    def log_summary(self, logger) -> None:
        """Print comprehensive summary."""
        logger.info("=" * 60)
        logger.info("MINING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Stage 1 Retrieved: {self.stage1_retrieved}")
        logger.info(f"  Filtered (Exact): {self.stage1_filtered_exact}")
        logger.info(f"  Filtered (>0.75 sim): {self.stage1_filtered_similar}")
        logger.info(f"  Valid Candidates: {self.stage1_after_filter}")
        logger.info(f"  Sampled Hard: {self.stage1_hard_sampled}")
        logger.info(f"  Sampled Medium: {self.stage1_medium_sampled}")
        logger.info(f"  Sampled Easy: {self.stage1_easy_sampled}")
        logger.info("-" * 60)
        logger.info(f"Stage 2 Candidates: {self.stage2_candidates}")
        logger.info(f"  Filtered: {self.stage2_filtered_similar}")
        logger.info(f"  Sampled: {self.stage2_sampled}")
        logger.info("-" * 60)
        logger.info(f"Total Before Dedup: {self.total_before_dedup}")
        logger.info(f"Total After Dedup: {self.total_after_dedup}")
        if self.fallback_sampled > 0:
            logger.info(f"Fallback Random Sampled: {self.fallback_sampled}")
        logger.info("-" * 60)
        logger.info(f"Avg Hard Similarity: {self.avg_hard_similarity:.3f}")
        logger.info(f"Avg Medium Similarity: {self.avg_medium_similarity:.3f}")
        logger.info(f"Avg Easy Similarity: {self.avg_easy_similarity:.3f}")
        logger.info(f"Avg Stage2 Similarity: {self.avg_stage2_similarity:.3f}")
        logger.info("=" * 60)
