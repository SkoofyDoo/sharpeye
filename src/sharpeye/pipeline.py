"""Pipeline orchestrator — single-frame image quality evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from sharpeye.config import Preset, load_preset
from sharpeye.gates.engine import evaluate_gates
from sharpeye.metrics.registry import compute_metrics, list_metrics
from sharpeye.preprocess import preprocess_frame
from sharpeye.report import BatchReport, FrameReport, Issue, build_human_summary
from sharpeye.scoring.classify import classify_by_percentile
from sharpeye.scoring.composite import composite_score
from sharpeye.scoring.normalize import normalize_metrics_batch


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
    
    def evaluate_batch(
        self, 
        images: list[str | Path | np.ndarray],
    ) -> BatchReport:
        frames: list[FrameReport] = []
        for image in images:
            frames.append(self.evaluate_frame(image))
        
        frames = self._apply_batch_scoring(frames)

        passed_count = sum(1 for f in frames if f.passed)
        failed_count = len(frames) - passed_count

        return BatchReport(
            preset = self.preset.name,
            total = len(frames),
            passed_count = passed_count,
            failed_count = failed_count,
            frames = frames,
        )
    
    def _apply_batch_scoring(self, frames: list[FrameReport]) -> list[FrameReport]:
        """Re-label non-catastrophic frames by composite percentile within batch"""   
        if len(frames) < 2:
            return frames
        
        weights = self.preset.scoring.weights
        
        if not weights:
            return frames

        metric_names = list(weights.keys())
        catastrophic_idx = {
            i for i, f in enumerate(frames)
            if any(iss.severity == "catastrophic" for iss in f.issues)
        }

        scoring_idx = [i for i in range(len(frames)) if i not in catastrophic_idx]
        if not scoring_idx:
            return frames
        
        metrics_subset = [frames[i].metrics for i in scoring_idx]
        normalized = normalize_metrics_batch(metrics_subset, metric_names)
        scores = [
            composite_score(norm, weights)
            for norm in normalized
        ]
        labels = classify_by_percentile(
            scores,
            self.preset.scoring.good_percentile,
            self.preset.scoring.medium_percentile,

        )
        updated = list(frames)
        for j, idx in enumerate(scoring_idx):
            label = labels[j]
            passed = label != "bad"
            score = scores[j]
            old = updated[idx]
            updated[idx] = replace(
                old,
                label = label,
                passed = passed,
                composite_score = score,
                human_summary = build_human_summary(old.issues, label)
            )
        return updated

def _classify(issues: list[Issue]) -> tuple[bool, str]:
    if any(i.severity == "catastrophic" for i in issues):
        return False, "bad"
    if issues:
        return True, "medium"
    return True, "good"