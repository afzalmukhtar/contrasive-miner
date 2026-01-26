"""
Tests for Contrastive Miner.
"""

import pytest
import numpy as np
import shutil
import tempfile
import os
import json
from typing import List, Dict, Set

from contrastive_miner.models import (
    TripletRow,
    MinerConfig,
    CandidateNegative,
    MiningStats,
)
from contrastive_miner.miner import SemanticNegativeMiner
from contrastive_miner.index import VectorIndex
from contrastive_miner.reranker import Reranker


# --- Mock Embedding Model ---


class MockEmbedder:
    """Mock embedding model for testing (SentenceTransformer-like interface)."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache = {}

    def encode(
        self,
        texts,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        device: str = "cpu",
    ) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = []
        for text in texts:
            if text not in self._cache:
                # Deterministic embedding based on text
                np.random.seed(hash(text) % 2**32)
                self._cache[text] = np.random.randn(self.dim).astype(np.float32)
            embeddings.append(self._cache[text])

        result = np.array(embeddings)
        return result[0] if single else result


def mock_embed_fn(texts):
    """Mock embedding function (callable interface)."""
    embedder = MockEmbedder(dim=64)
    return embedder.encode(texts)


def mock_rerank_fn(query: str, candidates: List[str]) -> List[tuple]:
    """Mock reranking function."""
    # Simple mock: return in reverse order with fake scores
    return [(c, 1.0 - i * 0.1) for i, c in enumerate(reversed(candidates))]


# --- Test Data ---


def get_test_data() -> List[Dict]:
    """Get sample test data."""
    return [
        {
            "anchor": "What is the revenue for 2025?",
            "positives": ["Revenue for 2025 was $5 billion."],
        },
        {"anchor": "Who is the CEO?", "positives": ["The CEO is Jane Doe."]},
        {
            "anchor": "What were the operating costs?",
            "positives": ["Operating costs were $2 billion."],
        },
        {"anchor": "What is EBITDA?", "positives": ["EBITDA was $1.5 billion."]},
        {
            "anchor": "What is the profit margin?",
            "positives": ["Profit margin was 15%."],
        },
    ]


# --- Tests for Models ---


class TestModels:
    def test_triplet_row_creation(self):
        """Test TripletRow creation with metadata."""
        row = TripletRow(
            anchor="query",
            positive="chunk",
            negatives=["neg1", "neg2"],
            negative_sources=["stage1_hard", "stage2"],
            negative_similarities=[0.8, 0.6],
        )

        assert row.anchor == "query"
        assert row.positive == "chunk"
        assert len(row.negatives) == 2
        assert len(row.negative_sources) == 2
        assert row.negative_similarities[0] == 0.8

    def test_triplet_to_dict(self):
        """Test TripletRow serialization."""
        row = TripletRow(
            anchor="query",
            positive="chunk",
            negatives=["neg1"],
            mining_stats=MiningStats(stage1_retrieved=10),
        )

        d = row.to_dict()
        assert d["anchor"] == "query"
        assert d["mining_stats"]["stage1_retrieved"] == 10

    def test_miner_config_defaults(self):
        """Test MinerConfig default values (stratified)."""
        config = MinerConfig()

        # Check new stratified defaults
        assert config.stage1_retrieve_k == 200
        assert config.stage1_similarity_threshold == 0.75
        assert config.stage1_hard_range == (0, 20)
        assert config.stage2_top_queries == 10
        assert config.multiplier == 1


# --- Tests for VectorIndex ---


class TestVectorIndex:
    def test_index_creation(self):
        """Test VectorIndex creation."""
        embeddings = np.random.randn(10, 128).astype(np.float32)
        metadata = [{"id": i} for i in range(10)]

        index = VectorIndex(embeddings, metadata, use_faiss=False)

        assert len(index) == 10

    def test_search_returns_top_k(self):
        """Test that search returns correct number of results."""
        embeddings = np.random.randn(20, 64).astype(np.float32)
        metadata = [{"id": i, "text": f"text_{i}"} for i in range(20)]

        index = VectorIndex(embeddings, metadata, use_faiss=False)

        query = embeddings[0]  # Search for first embedding
        results = index.search(query, top_k=5)

        assert len(results) == 5
        assert results[0]["id"] == 0  # Should find itself first

    def test_search_with_exclude(self):
        """Test that excluded indices are not returned."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        metadata = [{"id": i} for i in range(10)]

        index = VectorIndex(embeddings, metadata, use_faiss=False)

        # Exclude first 3 indices
        results = index.search(embeddings[0], top_k=5, exclude_indices={0, 1, 2})

        # Should not contain excluded indices
        result_ids = {r["id"] for r in results}
        assert 0 not in result_ids
        assert 1 not in result_ids
        assert 2 not in result_ids


# --- Tests for Miner ---


class TestSemanticNegativeMiner:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, "data.jsonl")

        # Save test data
        with open(self.data_path, "w") as f:
            for item in get_test_data():
                f.write(json.dumps(item) + "\n")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_miner_initialization(self):
        """Test miner initialization."""
        embedder = MockEmbedder()
        miner = SemanticNegativeMiner(embedder)
        assert miner.chunk_index is None

    def test_build_indices(self):
        """Test index building."""
        embedder = MockEmbedder()
        miner = SemanticNegativeMiner(embedder)

        data = get_test_data()
        miner.build_indices(data)

        assert miner.chunk_index is not None
        assert miner.query_index is not None
        assert len(miner.all_chunks) > 0

    def test_mine_dataset_e2e(self):
        """Test end-to-end mining."""
        embedder = MockEmbedder()
        config = MinerConfig(
            stage1_retrieve_k=5,
            stage1_hard_range=(0, 2),
            stage1_medium_range=(2, 4),
            stage1_easy_range=(4, 5),
            multiplier=1,
        )

        # Use mock reranker to avoid loading real models
        reranker = Reranker(rerank_fn=mock_rerank_fn)

        miner = SemanticNegativeMiner(embedder, config, reranker=reranker)

        data = get_test_data()
        triplets = miner.mine_dataset(data, log_stats=False)

        assert len(triplets) > 0
        assert isinstance(triplets[0], TripletRow)
        assert len(triplets[0].negatives) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
