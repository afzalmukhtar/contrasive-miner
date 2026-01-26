"""
Utility modules for contrastive miner.
"""

from .similarity import (
    cosine_similarity,
    batch_cosine_similarity,
    filter_by_similarity,
)

__all__ = [
    "cosine_similarity",
    "batch_cosine_similarity",
    "filter_by_similarity",
]
