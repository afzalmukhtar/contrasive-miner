"""
Contrastive Miner - Two-Stage Hard Negative Mining Library

A generalistic data preparation library that mines hard and random negatives
using a two-stage approach and produces datasets ready for Triplet Loss or
MNRL with explicit negatives training.

Usage:
    from sentence_transformers import SentenceTransformer
    from contrastive_miner import SemanticNegativeMiner, MinerConfig

    # Your embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure mining
    config = MinerConfig(
        stage1_retrieve_k=200,
        stage1_similarity_threshold=0.75,
        stage1_hard_range=(0, 20),
        multiplier=2
    )

    # Mine negatives (uses default cross-encoder for reranking)
    miner = SemanticNegativeMiner(embedder, config)
    triplets = miner.mine_dataset(data)

    # Or provide custom reranker
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    miner = SemanticNegativeMiner(embedder, config, cross_encoder=reranker)
"""

from .models import (
    TripletRow,
    MinerConfig,
    CandidateNegative,
    MiningStats,
)
from .miner import SemanticNegativeMiner
from .index import VectorIndex
from .reranker import Reranker
from .transforms import (
    load_data,
    save_data,
    normalize_input_data,
    group_by_anchor,
    explode_positives,
    sample_data,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "SemanticNegativeMiner",
    # Models
    "TripletRow",
    "MinerConfig",
    "CandidateNegative",
    "MiningStats",
    # Components
    "VectorIndex",
    "Reranker",
    # Utilities
    "load_data",
    "save_data",
    "normalize_input_data",
    "group_by_anchor",
    "explode_positives",
    "sample_data",
]
