"""Milestone 2.2 — pipeline integration tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sharpeye import Pipeline
from sharpeye.exceptions import InvalidImageError

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"


def _sharp_image(size: int = 200) -> np.ndarray:
    img = np.full((size, size), 80, dtype=np.uint8)
    for i in range(0, size, 10):
        cv2.line(img, (i, 0), (i, size), 255, 1)
    return img


def _blur_image(size: int = 200, ksize: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size), (ksize, ksize), 0)


def _dark_image(size: int = 200) -> np.ndarray:
    return np.full((size, size), 15, dtype=np.uint8)


@pytest.fixture
def pipe():
    return Pipeline.from_preset("default", presets_dir=PRESETS_DIR)


def test_sharp_image_passes(pipe):
    report = pipe.evaluate_frame(_sharp_image())
    assert report.passed is True
    assert report.label == "good"
    assert report.preset == "default"
    assert "laplacian_variance" in report.metrics
    assert report.issues == []


def test_dark_image_fails(pipe):
    report = pipe.evaluate_frame(_dark_image())
    assert report.passed is False
    assert report.label == "bad"
    assert any(i.code == "brightness_mean_too_low" for i in report.issues)
    assert "window" in report.human_summary.lower()


def test_blur_image_fails(pipe):
    report = pipe.evaluate_frame(_blur_image())
    assert report.passed is False
    assert any(i.metric == "laplacian_variance" for i in report.issues)


def test_human_summary_not_empty_on_failure(pipe):
    report = pipe.evaluate_frame(_dark_image())
    assert len(report.human_summary) > 10


def test_invalid_image_raises(pipe):
    with pytest.raises(InvalidImageError):
        pipe.evaluate_frame(np.array([]))


def test_frame_report_serializable(pipe):
    report = pipe.evaluate_frame(_sharp_image())
    d = report.to_dict()
    assert isinstance(d["metrics"], dict)
    assert isinstance(d["issues"], list)
    assert d["preset"] == "default"


def test_default_preset_computes_all_six_metrics(pipe):
    report = pipe.evaluate_frame(_sharp_image())
    expected = {
        "laplacian_variance",
        "brightness_mean",
        "contrast_std",
        "tenengrad",
        "noise_std",
        "edge_laplacian_p90",
    }
    assert set(report.metrics.keys()) == expected


def test_telemedicine_preset_loads_and_evaluates():
    pipe = Pipeline.from_preset("telemedicine", presets_dir=PRESETS_DIR)
    report = pipe.evaluate_frame(_dark_image())
    assert report.preset == "telemedicine"
    assert report.passed is False
    assert "window" in report.human_summary.lower()


def _low_contrast_sharp_image(size: int = 200) -> np.ndarray:
    img = np.full((size, size), 127, dtype=np.uint8)
    for i in range(0, size, 4):
        cv2.line(img, (i, 0), (i, size), 133, 1)
    return img


def test_telemedicine_soft_contrast_issue():
    pipe = Pipeline.from_preset("telemedicine", presets_dir=PRESETS_DIR)
    report = pipe.evaluate_frame(_low_contrast_sharp_image())
    contrast_issues = [i for i in report.issues if i.metric == "contrast_std"]
    assert contrast_issues
    assert contrast_issues[0].severity == "soft"
    assert report.passed is True
    assert report.label == "medium"
    assert "contrast" in report.human_summary.lower()