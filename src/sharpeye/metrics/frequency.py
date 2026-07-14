"""Frequency-domain metrics — opt-in for dataset_cleaner."""

from __future__ import annotations

import numpy as np

from sharpeye.metrics.base import Metric


class HfEnergyRatio(Metric):
    """Share of spectral energy outside the low-frequency center (JPEG artifact proxy)."""

    name = "hf_energy_ratio"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
        power = np.abs(spectrum) ** 2
        h, w = power.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * 0.15)
        y, x = np.ogrid[:h, :w]
        low_mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        total = float(power.sum())
        if total <= 0:
            return 0.0
        low_energy = float(power[low_mask].sum())
        return float(1.0 - low_energy / total)