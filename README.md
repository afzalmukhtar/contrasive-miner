# Contrastive Miner

A **two-stage hard negative mining** library for contrastive learning that produces high-quality training datasets for Triplet Loss or MNRL.

## Features

- **Stage 1 (Document Retrieval)**: Finds chunks that semantically confuse the model
- **Stage 2 (Query Similarity)**: Finds chunks from similar queries as negatives
- **Re-ranking**: Optional cross-encoder support for better hard negatives
- **Intermediate Output**: Stage-specific results for analysis
- **Final Output**: Flattened format ready for training

## Installation

```bash
cd contrastive-miner
pip install -e .

# With FAISS support (faster)
pip install -e ".[faiss]"
```

## Quick Start

### Python API

```python
from sentence_transformers import SentenceTransformer
from contrastive_miner import SemanticNegativeMiner, MinerConfig

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configure mining
config = MinerConfig(
    stage1_n_hard=2,
    stage1_n_random=1,
    stage2_top_queries=10,
    stage2_n_hard=2,
    stage2_n_random=1
)

# Create miner
miner = SemanticNegativeMiner(model, config)

# Mine dataset
data = [
    {"anchor": "What is revenue?", "positives": ["Revenue is $5B..."]},
    {"anchor": "Who is the CEO?", "positives": ["The CEO is..."]},
]

intermediate, triplets = miner.mine_dataset(
    data,
    intermediate_path="intermediate.jsonl",
    final_path="triplets.jsonl"
)
```

### CLI

```bash
python -m contrastive_miner \
    --input train.json \
    --output triplets.jsonl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --stage1-n-hard 2 \
    --stage1-n-random 1 \
    --stage2-n-hard 2 \
    --stage2-n-random 1 \
    --intermediate intermediate.jsonl \
    --group-by-anchor
```

## Input Format

Supports multiple formats:
```json
{"anchor": "query", "positives": ["chunk1", "chunk2"]}
{"query": "query", "positive": "chunk"}
```

Use `--group-by-anchor` to combine repeated queries.

## Output Format

### Intermediate (Stage-specific)
```json
{
    "anchor": "query",
    "positive": "chunk",
    "hard_neg_doc": ["from stage 1"],
    "random_neg_doc": ["from stage 1"],
    "hard_neg_query": ["from stage 2"],
    "random_neg_query": ["from stage 2"]
}
```

### Final (Flattened)
```json
{
    "anchor": "query",
    "positive": "chunk",
    "negatives": ["all negatives combined"]
}
```

## Algorithm

### Stage 1: Document Retrieval
1. Retrieve top-K chunks for query
2. Filter out positive chunks
3. Re-rank remaining by relevance
4. Select top-n as hard negatives
5. Sample random chunks as random negatives

### Stage 2: Query Similarity
1. Find top-Y similar queries
2. Collect their positives as candidates
3. Filter out original positives
4. Re-rank by original query
5. Select top-n as hard negatives
6. Sample random as random negatives

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage1_n_hard` | 1 | Hard negatives from doc retrieval |
| `stage1_n_random` | 1 | Random negatives from doc retrieval |
| `stage1_retrieve_buffer` | 10 | Extra chunks to retrieve |
| `stage2_top_queries` | 10 | Similar queries to retrieve |
| `stage2_n_hard` | 1 | Hard negatives from query similarity |
| `stage2_n_random` | 1 | Random negatives from query similarity |

## 🚀 Case Study: Financial Domain Adaptation

Using `contrastive-miner` to train a simple **Linear Adapter** on financial data yielded projected performance gains compared to standard random negative sampling.

**Impact on Downstream Retrieval (NDCG@10):**

| Method | NDCG@10 | Recall@10 | Improvement |
|--------|---------|-----------|-------------|
| Random Sampling (Baseline) | 0.379 | 57.6% | - |
| **Contrastive Miner** | **0.460** | **67.2%** | **+21%** |

> "The better quality of hard negatives allowed a simple linear layer to learn robust domain-specific distinctions that previously required full model fine-tuning."

🔴 **Verify these results:**
Check out the full implementation and benchmarks in the [Fine-Tuning Embeddings on Domain-Specific Data](https://github.com/afzalmukhtar/fine-tuning-embeddings-on-domain-specific-data) repository.

## License

MIT
