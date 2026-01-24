"""
Data models for Contrastive Miner.

Defines the data structures for:
- IntermediateRow: Stores stage-specific negatives
- TripletRow: Final flattened format for training
- MinerConfig: Configuration for the mining process
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Set
from pydantic import BaseModel, Field
import json


@dataclass
class IntermediateRow:
    """
    Intermediate format that preserves stage-specific negatives.

    This is saved first for debugging and analysis before
    flattening to the final TripletRow format.
    """

    anchor: str
    positive: str

    # Stage 1: Document Retrieval Negatives
    hard_neg_doc: List[str] = field(default_factory=list)
    random_neg_doc: List[str] = field(default_factory=list)

    # Stage 2: Query Similarity Negatives
    hard_neg_query: List[str] = field(default_factory=list)
    random_neg_query: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntermediateRow":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TripletRow:
    """
    Final flattened format for Triplet Loss training.

    Each row has one positive and multiple negatives that can be
    used with MultipleNegativesRankingLoss or TripletLoss.
    """

    anchor: str
    positive: str
    negatives: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TripletRow":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_intermediate(cls, intermediate: IntermediateRow) -> "TripletRow":
        """
        Flatten an IntermediateRow to TripletRow.

        Combines all negatives from both stages into a single list.
        """
        all_negatives = (
            intermediate.hard_neg_doc
            + intermediate.random_neg_doc
            + intermediate.hard_neg_query
            + intermediate.random_neg_query
        )
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_negatives = []
        for neg in all_negatives:
            if neg not in seen:
                seen.add(neg)
                unique_negatives.append(neg)

        return cls(
            anchor=intermediate.anchor,
            positive=intermediate.positive,
            negatives=unique_negatives,
        )


class MinerConfig(BaseModel):
    """Configuration for the SemanticNegativeMiner."""

    # Stage 1: Document Retrieval config
    stage1_n_hard: int = Field(
        default=1, description="Number of hard negatives from document retrieval"
    )
    stage1_n_random: int = Field(
        default=1, description="Number of random negatives from document retrieval"
    )
    stage1_retrieve_buffer: int = Field(
        default=10, description="Extra chunks to retrieve for filtering buffer"
    )

    # Stage 2: Query Similarity config
    stage2_top_queries: int = Field(
        default=10, description="Number of similar queries to retrieve"
    )
    stage2_n_hard: int = Field(
        default=1, description="Number of hard negatives from query similarity"
    )
    stage2_n_random: int = Field(
        default=1, description="Number of random negatives from query similarity"
    )

    # General config
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for similarity computation",
    )
    use_reranker: bool = Field(
        default=False, description="Whether to use cross-encoder for re-ranking"
    )
    reranker_model: Optional[str] = Field(
        default=None, description="Cross-encoder model for re-ranking"
    )
    batch_size: int = Field(
        default=32, description="Batch size for embedding computation"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()


def save_intermediate(rows: List[IntermediateRow], path: str) -> None:
    """Save intermediate rows to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def load_intermediate(path: str) -> List[IntermediateRow]:
    """Load intermediate rows from JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(IntermediateRow.from_dict(json.loads(line)))
    return rows


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


def flatten_intermediate_to_triplets(
    intermediate_rows: List[IntermediateRow],
) -> List[TripletRow]:
    """Convert all intermediate rows to triplet rows."""
    return [TripletRow.from_intermediate(row) for row in intermediate_rows]
