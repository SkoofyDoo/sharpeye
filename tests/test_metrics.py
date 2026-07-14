"""Milestone 1.3 — metric plugin tests."""

import cv2
import numpy as np
import pytest

from sharpeye.exceptions import MetricError
from sharpeye.metrics.registry import compute_metrics, get_metric, list_metrics


def _sharp_image(size: int = 200) -> np.ndarray:
    img = np.full((size, size), 80, dtype=np.uint8)
    for i in range(0, size, 10):
        cv2.line(img, (i, 0), (i, size), 255, 1)
    return img


def _blur_image(size: int = 200, ksize: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size), (ksize, ksize), 0)


def _dark_image(size: int = 200) -> np.ndarray:
    return np.full((size, size), 15, dtype=np.uint8)


def test_registry_lists_builtins():
    names = list_metrics()
    assert "laplacian_variance" in names
    assert "brightness_mean" in names
    assert "contrast_std" in names


def test_unknown_metric_raises():
    with pytest.raises(MetricError):
        get_metric("nonexistent")


def test_laplacian_sharp_greater_than_blur():
    sharp = _sharp_image()
    blur = _blur_image()
    ctx: dict = {}
    metrics = compute_metrics(sharp, ["laplacian_variance"], ctx)
    blur_metrics = compute_metrics(blur, ["laplacian_variance"], {})
    assert metrics["laplacian_variance"] > blur_metrics["laplacian_variance"]


def test_brightness_dark_lower_than_sharp():
    dark = _dark_image()
    sharp = _sharp_image()
    metrics = compute_metrics(dark, ["brightness_mean"], {})
    sharp_metrics = compute_metrics(sharp, ["brightness_mean"], {})
    assert metrics["brightness_mean"] < sharp_metrics["brightness_mean"]


def test_contrast_sharp_greater_than_flat():
    sharp = _sharp_image()
    flat = np.full((200, 200), 128, dtype=np.uint8)
    sharp_metrics = compute_metrics(sharp, ["contrast_std"], {})
    flat_metrics = compute_metrics(flat, ["contrast_std"], {})
    assert sharp_metrics["contrast_std"] > flat_metrics["contrast_std"]


def test_ctx_cache_reuses_laplacian():
    gray = _sharp_image()
    ctx: dict = {}
    compute_metrics(gray, ["laplacian_variance"], ctx)
    assert "laplacian" in ctx
    lap_first = ctx["laplacian"]
    compute_metrics(gray, ["laplacian_variance"], ctx)
    assert ctx["laplacian"] is lap_first


def test_compute_multiple_metrics():
    gray = _sharp_image()
    result = compute_metrics(
        gray,
        ["laplacian_variance", "brightness_mean", "contrast_std"],
        {},
    )
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result.values())