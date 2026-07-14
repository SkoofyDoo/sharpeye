"""Min-max normalization of metrics across a batch."""

from __future__ import annotations

_HIGHER_IS_BETTER = frozenset({
    "laplacian_variance", "tenengrad", "contrast_std", "hf_energy_ratio",
})
_LOWER_IS_BETTER = frozenset({"noise_std"})


def normalize_metrics_batch(
    metrics_list: list[dict[str, float]],
    metric_names: list[str],
) -> list[dict[str, float]]:
    """Normalize each metric to [0, 1] across the batch."""
    if not metrics_list:
        return []

    n = len(metrics_list)
    result: list[dict[str, float]] = [{} for _ in range(n)]

    for name in metric_names:
        values = [m.get(name, 0.0) for m in metrics_list]
        vmin = min(values)
        vmax = max(values)

        for i, raw in enumerate(values):
            if vmax == vmin:
                norm = 1.0
            elif name in _LOWER_IS_BETTER:
                norm = (vmax - raw) / (vmax - vmin)
            else:
                norm = (raw - vmin) / (vmax - vmin)
            result[i][name] = float(norm)

    return result