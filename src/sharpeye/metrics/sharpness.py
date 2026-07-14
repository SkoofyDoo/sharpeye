"""Sharpness-related metrics."""

from __future__ import annotations

import cv2
import numpy as np

from sharpeye.metrics.base import Metric


class LaplacianVariance(Metric):
    """ Sum of squared Sobel gradient magnitudes - blur-sensitive sharpness """
    name = "Laplacian_Variance"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        lap = ctx.get("laplacian")
        if lap is None:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            ctx["laplacian"] = lap
        return float(lap.var())

class Tenengrad(Metric):
    name = "Tenengrad"
    
    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        gx = ctx.get("sobel_x")
        gy = ctx.get("sobel_y")
        if gx is None or gy is None:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)
            ctx["sobel_x"] = gx
            ctx["sobel_y"] = gy
        magnitude_sq = gx * gx + gy * gy
        return float(np.mean(magnitude_sq))

