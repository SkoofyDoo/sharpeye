"""Percentile-based label assigment for batch scoring"""

from __future__ import annotations

import numpy as np


def classify_by_percentile(
    scores: list[float],
    good_percentile: float,
    medium_percentile: float
) -> list[str]:
    """ Assign good / medium / bad by composite score rank in batch.
        good_percentile = 0.67 -> score >= 67th pct -> good
        medium_percentile = 0.33 -> score >= 33rd pct -> medium
    """

    if not scores:
        return []
    
    arr = np.array(scores, dtype = np.float64)
    good_thresh = float(np.percentile(arr, good_percentile * 100))
    medium_thresh = float(np.percentile(arr, medium_percentile * 100))

    labels: list[str] = []
    for s in scores:
        if s >= good_thresh:
            labels.append("good")
        elif s >= medium_thresh:
            labels.append("medium")
        else:
            labels.append("bad")

    return labels