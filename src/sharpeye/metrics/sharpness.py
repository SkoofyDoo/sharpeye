"""Sharpness-related metrics."""

from __future__ import annotations

import cv2
import numpy as np

from sharpeye.metrics.base import Metric


class LaplacianVariance(Metric):
    name = "laplacian_variance"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        lap = ctx.get("laplacian")
        if lap is None:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            ctx["laplacian"] = lap
        return float(lap.var())