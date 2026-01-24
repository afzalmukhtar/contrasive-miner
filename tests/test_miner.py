"""
Tests for Contrastive Miner.
"""

import pytest
import numpy as np
from typing import List, Dict

from contrastive_miner.models import (
    IntermediateRow,
    TripletRow,
    MinerConfig,
    flatten_intermediate_to_triplets,
)
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
    def test_intermediate_row_creation(self):
        """Test IntermediateRow creation."""
        row = IntermediateRow(
            anchor="query",
            positive="chunk",
            hard_neg_doc=["hard1"],
            random_neg_doc=["random1"],
            hard_neg_query=["hard2"],
            random_neg_query=["random2"],
        )

        assert row.anchor == "query"
        assert row.positive == "chunk"
        assert len(row.hard_neg_doc) == 1

    def test_intermediate_to_dict(self):
        """Test IntermediateRow serialization."""
        row = IntermediateRow(anchor="query", positive="chunk")

        d = row.to_dict()
        assert d["anchor"] == "query"
        assert d["positive"] == "chunk"

    def test_triplet_from_intermediate(self):
        """Test flattening IntermediateRow to TripletRow."""
        intermediate = IntermediateRow(
            anchor="query",
            positive="chunk",
            hard_neg_doc=["neg1"],
            random_neg_doc=["neg2"],
            hard_neg_query=["neg3"],
            random_neg_query=["neg4"],
        )

        triplet = TripletRow.from_intermediate(intermediate)

        assert triplet.anchor == "query"
        assert triplet.positive == "chunk"
        assert len(triplet.negatives) == 4
        assert "neg1" in triplet.negatives
        assert "neg4" in triplet.negatives

    def test_triplet_deduplicates_negatives(self):
        """Test that duplicate negatives are removed."""
        intermediate = IntermediateRow(
            anchor="query",
            positive="chunk",
            hard_neg_doc=["neg1", "neg2"],
            random_neg_doc=["neg2", "neg3"],  # neg2 is duplicate
            hard_neg_query=["neg3"],  # neg3 is duplicate
            random_neg_query=["neg4"],
        )

        triplet = TripletRow.from_intermediate(intermediate)

        # Should have 4 unique negatives, not 6
        assert len(triplet.negatives) == 4
        assert triplet.negatives == ["neg1", "neg2", "neg3", "neg4"]

    def test_miner_config_defaults(self):
        """Test MinerConfig default values."""
        config = MinerConfig()

        assert config.stage1_n_hard == 1
        assert config.stage1_n_random == 1
        assert config.stage2_n_hard == 1


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

    def test_get_random_indices(self):
        """Test random index sampling."""
        embeddings = np.random.randn(100, 64).astype(np.float32)
        metadata = [{"id": i} for i in range(100)]

        index = VectorIndex(embeddings, metadata, use_faiss=False)

        random_indices = index.get_random_indices(5, exclude_indices={0, 1, 2})

        assert len(random_indices) == 5
        assert 0 not in random_indices
        assert 1 not in random_indices


# --- Tests for Reranker ---


class TestReranker:
    def test_reranker_with_custom_function(self):
        """Test reranker with custom reranking function."""
        reranker = Reranker(rerank_fn=mock_rerank_fn)

        query = "test query"
        candidates = ["a", "b", "c"]

        results = reranker.rerank(query, candidates)

        assert len(results) == 3
        # Our mock reverses the order
        assert results[0][0] == "c"

    def test_get_top_texts(self):
        """Test getting only top text results."""
        reranker = Reranker(rerank_fn=mock_rerank_fn)

        query = "test query"
        candidates = ["a", "b", "c", "d", "e"]

        top_texts = reranker.get_top_texts(query, candidates, top_n=3)

        assert len(top_texts) == 3
        assert all(isinstance(t, str) for t in top_texts)

    def test_reranker_empty_candidates(self):
        """Test reranker with empty candidates list."""
        reranker = Reranker(rerank_fn=mock_rerank_fn)

        results = reranker.rerank("query", [])
        assert results == []


# --- Tests for Flatten ---


class TestFlatten:
    def test_flatten_intermediate_to_triplets(self):
        """Test batch flattening of intermediate rows."""
        intermediates = [
            IntermediateRow(
                anchor="q1", positive="p1", hard_neg_doc=["n1"], random_neg_doc=["n2"]
            ),
            IntermediateRow(
                anchor="q2",
                positive="p2",
                hard_neg_query=["n3"],
                random_neg_query=["n4"],
            ),
        ]

        triplets = flatten_intermediate_to_triplets(intermediates)

        assert len(triplets) == 2
        assert triplets[0].anchor == "q1"
        assert triplets[1].anchor == "q2"


# --- Tests for Embedder Interface ---


class TestEmbedderInterface:
    def test_sentence_transformer_like_interface(self):
        """Test that MockEmbedder works like SentenceTransformer."""
        embedder = MockEmbedder(dim=64)

        # Single text
        single = embedder.encode("test")
        assert single.shape == (64,)

        # Multiple texts
        batch = embedder.encode(["a", "b", "c"])
        assert batch.shape == (3, 64)

    def test_callable_embedder_interface(self):
        """Test that callable function works as embedder."""
        result = mock_embed_fn("test")
        assert isinstance(result, np.ndarray)


# --- Run with pytest ---

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
