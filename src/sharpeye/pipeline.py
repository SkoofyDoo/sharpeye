"""Pipeline orchestrator — single-frame image quality evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sharpeye.config import Preset, load_preset
from sharpeye.gates.engine import evaluate_gates
from sharpeye.metrics.registry import compute_metrics, list_metrics
from sharpeye.preprocess import preprocess_frame
from sharpeye.report import FrameReport, Issue, build_human_summary


class Pipeline:
    """End-to-end IQC pipeline for a single frame."""

    def __init__(self, preset: Preset) -> None:
        self.preset = preset

    @classmethod
    def from_preset(
        cls,
        name: str,
        presets_dir: Path | None = None,
    ) -> Pipeline:
        return cls(load_preset(name, presets_dir = presets_dir))

    def _resolve_metrics(self) -> list[str]:
        available = set(list_metrics())
        return [m for m in self.preset.enabled_metrics if m in available]

    def evaluate_frame(self, image: str | Path | np.ndarray) -> FrameReport:
        gray, ctx = preprocess_frame(image, self.preset.preprocess)
        ctx["noise_kernel_size"] = self.preset.metrics.noise_kernel_size

        metrics = compute_metrics(gray, self._resolve_metrics(), ctx)

        issues = evaluate_gates(
            metrics,
            self.preset.gates.rules,
            self.preset.gates.groups,
        )

        passed, label = _classify(issues)
        human_summary = build_human_summary(issues, label)

        return FrameReport(
            passed = passed,
            label = label,
            metrics = metrics,
            issues = issues,
            human_summary = human_summary,
            preset = self.preset.name,
        )


def _classify(issues: list[Issue]) -> tuple[bool, str]:
    if any(i.severity == "catastrophic" for i in issues):
        return False, "bad"
    if issues:
        return True, "medium"
    return True, "good"