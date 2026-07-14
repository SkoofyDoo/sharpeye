"""Milestone 1.2 — contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sharpeye.config import Preset, list_presets, load_preset
from sharpeye.exceptions import PresetNotFoundError
from sharpeye.report import FrameReport, Issue, build_human_summary

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"


def test_load_default_preset():
    preset = load_preset("default", presets_dir=PRESETS_DIR)
    assert preset.name == "default"
    assert preset.preprocess.max_width == 640
    assert len(preset.gates.rules) >= 1


def test_preset_weights_sum():
    preset = load_preset("default", presets_dir=PRESETS_DIR)
    total = sum(preset.scoring.weights.values())
    assert abs(total - 1.0) < 0.01


def test_preset_not_found():
    with pytest.raises(PresetNotFoundError):
        load_preset("nonexistent", presets_dir=PRESETS_DIR)


def test_list_presets():
    names = list_presets(presets_dir=PRESETS_DIR)
    assert "default" in names
    assert "telemedicine" in names


def test_load_telemedicine_preset():
    preset = load_preset("telemedicine", presets_dir=PRESETS_DIR)
    assert preset.name == "telemedicine"
    assert preset.metrics.noise_kernel_size == 7
    assert "tenengrad" in preset.enabled_metrics
    assert len(preset.gates.rules) == 3
    assert len(preset.gates.groups) == 1
    total = sum(preset.scoring.weights.values())
    assert abs(total - 1.0) < 0.01


def test_frame_report_to_dict_no_numpy():
    report = FrameReport(
        passed=False,
        label="bad",
        metrics={"laplacian_variance": 12.3456},
        issues=[
            Issue(
                code="too_blurry",
                severity="catastrophic",
                message="Image appears blurry.",
                suggestion="Hold the camera steady.",
                metric="laplacian_variance",
                value=12.3,
            )
        ],
        human_summary="Image appears blurry. Suggestion: Hold the camera steady.",
        preset="default",
    )
    d = report.to_dict()
    json.dumps(d)  # must not raise
    assert d["passed"] is False
    assert d["metrics"]["laplacian_variance"] == 12.3456
    assert d["composite_score"] is None


def test_build_human_summary_with_issues():
    issues = [
        Issue(
            code="too_dark",
            severity="catastrophic",
            message="Image is too dark.",
            suggestion="Move closer to a window.",
        )
    ]
    summary = build_human_summary(issues, label="bad")
    assert "too dark" in summary.lower()
    assert "window" in summary.lower()


def test_invalid_weights_raise():
    with pytest.raises(ValidationError):
        Preset.model_validate({
            "name": "bad",
            "scoring": {"weights": {"a": 0.3, "b": 0.3}},  # sum 0.6
        })