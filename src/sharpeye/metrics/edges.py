"""Edge-based metrics — adaptive Canny on ROI."""

from __future__ import annotations

import cv2
import numpy as np

from sharpeye.metrics.base import Metric


def _adaptive_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds from image median (MASTER_PLAN 1.4)."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


class EdgeLaplacianP90(Metric):
    """90th percentile of Laplacian on ROI edge pixels."""

    name = "edge_laplacian_p90"

    def compute(self, gray: np.ndarray, ctx: dict) -> float:
        roi = ctx.get("roi_gray", gray)

        edges = ctx.get("canny_edges")
        if edges is None:
            edges = _adaptive_canny(roi)
            ctx["canny_edges"] = edges

        lap_roi = cv2.Laplacian(roi, cv2.CV_64F)
        edge_mask = edges > 0
        if not np.any(edge_mask):
            return float(np.percentile(np.abs(lap_roi), 90))

        values = np.abs(lap_roi)[edge_mask]
        return float(np.percentile(values, 90))