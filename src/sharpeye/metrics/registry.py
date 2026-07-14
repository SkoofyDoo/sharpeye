"""Metric plugin registry."""

from __future__ import annotations

import numpy as np

from sharpeye.exceptions import MetricError
from sharpeye.metrics.base import Metric
from sharpeye.metrics.exposure import BrightnessMean, ContrastStd
from sharpeye.metrics.sharpness import LaplacianVariance

_REGISTRY: dict[str, Metric] = {}


def register_metric(metric: Metric) -> None:
    _REGISTRY[metric.name] = metric


def get_metric(name: str) -> Metric:
    if name not in _REGISTRY:
        raise MetricError(f"Unknown metric: {name}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    return sorted(_REGISTRY.keys())


def compute_metrics(
    gray: np.ndarray,
    names: list[str],
    ctx: dict | None = None,
) -> dict[str, float]:
    ctx = ctx if ctx is not None else {}
    results: dict[str, float] = {}
    for name in names:
        metric = get_metric(name)
        results[name] = float(metric.compute(gray, ctx))
    return results


def _register_builtins() -> None:
    for metric in (LaplacianVariance(), BrightnessMean(), ContrastStd()):
        register_metric(metric)


_register_builtins()