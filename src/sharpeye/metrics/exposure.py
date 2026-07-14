"""Exposure and contrast metrics."""

from __future__ import annotations

import numpy as np

from sharpeye.metrics.base import Metric


class BrightnessMean(Metric):
    name = "brightness_mean"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        return float(np.mean(gray))


class ContrastStd(Metric):
    name = "contrast_std"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        return float(np.std(gray))