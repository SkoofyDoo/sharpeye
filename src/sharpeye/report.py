"""Report models — output contract for frames and batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    suggestion: str
    metric: str | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "metric": self.metric,
            "value": self.value,
        }


@dataclass
class FrameReport:
    passed: bool
    label: str
    metrics: dict[str, float]
    issues: list[Issue] = field(default_factory=list)
    human_summary: str = ""
    preset: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "label": self.label,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "issues": [i.to_dict() for i in self.issues],
            "human_summary": self.human_summary,
            "preset": self.preset,
        }


@dataclass
class BatchReport:
    preset: str
    total: int
    passed_count: int
    failed_count: int
    frames: list[FrameReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "frames": [f.to_dict() for f in self.frames],
        }


def build_human_summary(issues: list[Issue], label: str) -> str:
    if not issues:
        if label == "good":
            return "Image quality looks good."
        if label == "medium":
            return "Image quality is acceptable with minor issues."
        return "Image quality is poor."

    catastrophic = [i for i in issues if i.severity == "catastrophic"]
    target = catastrophic if catastrophic else issues

    parts = [i.message for i in target[:2]]
    summary = ". ".join(parts)

    suggestions = [i.suggestion for i in target if i.suggestion]
    if suggestions:
        summary += f" Suggestion: {suggestions[0]}"

    return summary