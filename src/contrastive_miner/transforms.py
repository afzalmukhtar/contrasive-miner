"""
Data transformation utilities.

Handles loading, saving, and transforming data between formats.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSON or JSONL file.

    Supports both formats automatically based on extension.

    Args:
        path: Path to data file

    Returns:
        List of data dictionaries
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def save_data(data: List[Dict[str, Any]], path: str, format: str = "jsonl") -> None:
    """
    Save data to JSON or JSONL file.

    Args:
        data: List of dictionaries to save
        path: Output path
        format: 'json' or 'jsonl'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if format == "jsonl":
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_input_data(data: List[Dict]) -> List[Dict]:
    """
    Normalize input data to standard format.

    Converts various input formats to:
    {"anchor": str, "positives": List[str]}

    Supported formats:
    - {"anchor": "...", "positives": [...]}
    - {"query": "...", "positive": "..."}
    - {"query": "...", "positives": [...]}

    Args:
        data: Input data in various formats

    Returns:
        Normalized data
    """
    normalized = []

    for item in data:
        anchor = item.get("anchor") or item.get("query")
        positives = item.get("positives")

        if positives is None:
            positive = item.get("positive")
            positives = [positive] if positive else []

        if anchor and positives:
            if isinstance(positives, str):
                positives = [positives]

            normalized.append({"anchor": anchor, "positives": positives})

    return normalized


def group_by_anchor(data: List[Dict]) -> List[Dict]:
    """
    Group data by anchor, collecting all positives.

    Useful when input has one row per (query, positive) pair
    and you want to combine into (query, [positives]).

    Args:
        data: Input data with potentially repeated anchors

    Returns:
        Grouped data with unique anchors and collected positives
    """
    anchor_to_positives: Dict[str, set] = {}

    for item in normalize_input_data(data):
        anchor = item["anchor"]

        if anchor not in anchor_to_positives:
            anchor_to_positives[anchor] = set()

        anchor_to_positives[anchor].update(item["positives"])

    return [
        {"anchor": anchor, "positives": list(positives)}
        for anchor, positives in anchor_to_positives.items()
    ]


def explode_positives(data: List[Dict]) -> List[Dict]:
    """
    Explode data so each positive has its own row.

    Converts {"anchor": "...", "positives": ["a", "b"]}
    to [{"anchor": "...", "positive": "a"}, {"anchor": "...", "positive": "b"}]

    Args:
        data: Input data with lists of positives

    Returns:
        Exploded data with one row per positive
    """
    normalized = normalize_input_data(data)
    exploded = []

    for item in normalized:
        for positive in item["positives"]:
            exploded.append({"anchor": item["anchor"], "positive": positive})

    return exploded


def sample_data(data: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    Sample n items from data.

    Args:
        data: Input data
        n: Number of items to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled subset
    """
    import random

    random.seed(seed)

    if n >= len(data):
        return data

    return random.sample(data, n)
