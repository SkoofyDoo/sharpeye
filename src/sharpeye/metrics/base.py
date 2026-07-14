"""Abstract base class for metric plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """A single image quality metric computed on grayscale input."""

    name: str

    @abstractmethod
    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        """Compute metric value from grayscale image and shared context cache."""