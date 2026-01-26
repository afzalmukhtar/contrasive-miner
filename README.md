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

# Configure mining (Stratified Sampling)
config = MinerConfig(
    stage1_retrieve_k=200,
    stage1_hard_range=(0, 20),
    hard_multiplier=2,
    stage2_top_queries=10,
    stage2_multiplier=2
)

# Create miner
miner = SemanticNegativeMiner(model, config)

# Mine dataset
data = [
    {"anchor": "What is revenue?", "positives": ["Revenue is $5B..."]},
    {"anchor": "Who is the CEO?", "positives": ["The CEO is..."]},
]

# Returns List[TripletRow]
triplets = miner.mine_dataset(
    data,
    output_path="triplets.jsonl"
)
```

## Input Format

Supports multiple formats:
```json
{"anchor": "query", "positives": ["chunk1", "chunk2"]}
{"query": "query", "positive": "chunk"}
```


## Output Format


### Final (Flattened)
```json
{
    "anchor": "query",
    "positive": "chunk",
    "negatives": ["negative1", "negative2", ...],
    "negative_sources": ["stage1_hard", "stage2_topic_neighbor", ...],
    "negative_similarities": [0.85, 0.42, ...]
}
```

## Algorithm

### Stage 1: Document Retrieval (Stratified)
1. Retrieve top-K chunks for query
2. Filter out positive chunks & high similarity false positives
3. Re-rank remaining by relevance with Cross-Encoder
4. Bucket into **Hard**, **Medium**, and **Easy** tiers
5. Sample negatives from each bucket based on multipliers

### Stage 2: Topic Neighbor Mining
1. Find top-Y similar queries
2. Collect their positives as candidates
3. Filter out original positives
4. Re-rank by original query
5. Select negatives based on `stage2_multiplier`

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage1_retrieve_k` | 200 | Initial candidates to retrieve |
| `stage1_hard_range` | (0, 20) | Rank range for "Hard" negatives |
| `stage1_medium_range` | (20, 50) | Rank range for "Medium" negatives |
| `stage1_easy_range` | (50, 200) | Rank range for "Easy" negatives |
| `hard_multiplier` | 2 | Negatives to sample from Hard bucket |
| `medium_multiplier` | 2 | Negatives to sample from Medium bucket |
| `easy_multiplier` | 2 | Negatives to sample from Easy bucket |
| `stage2_top_queries` | 10 | Similar queries to explore |
| `stage2_multiplier` | 2 | Negatives to sample from Stage 2 |
| `cross_encoder_model` | ms-marco.. | Model used for re-ranking |
| `use_true_random_easy` | False | Use true random negatives instead of ranks 50-200 |
| `true_random_similarity_threshold` | 0.3 | Max similarity for true random negatives |

### When to Use True Random Negatives

By default, the "Easy" bucket draws from ranks 50-200, which are **semi-hard negatives** (still top retrieved results sharing keywords/concepts with the query).

| Loss Function | Recommended Setting | Reason |
|---------------|---------------------|--------|
| **MNRL** (batch > 32) | `use_true_random_easy=False` (default) | Other rows in the batch provide the "true random" signal |
| **TripletLoss** (without in-batch negatives) | `use_true_random_easy=True` | Need explicit true random negatives |

```python
# For TripletLoss without in-batch negatives
config = MinerConfig(
    use_true_random_easy=True,
    true_random_similarity_threshold=0.3,  # Lower = more dissimilar
)
```

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
