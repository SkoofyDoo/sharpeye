"""Noise-related metrics."""

from __future__ import annotations

import cv2
import numpy as np

from sharpeye.metrics.base import Metric


class NoiseStd(Metric):
    """High-frequency noise estimate: std(gray - gaussian_blur)."""

    name = "noise_std"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        k = int(ctx.get("noise_kernel_size", 7))
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        residual = gray.astype(np.float64) - blurred.astype(np.float64)
        return float(np.std(residual))