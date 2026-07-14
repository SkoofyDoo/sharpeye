"""Weighted composite quality score."""

from __future__ import annotations


def composite_score(
    normalized: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted sum of normalized metrics (weights from preset YAML)."""
    total = 0.0
    for name, weight in weights.items():
        total += normalized.get(name, 0.0) * weight
    return float(total)