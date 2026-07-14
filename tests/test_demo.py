"""Smoke tests for Gradio demo."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest


def _sharp_rgb(size: int = 200) -> np.ndarray:
    gray = np.full((size, size), 80, dtype = np.uint8)
    for i in range(0, size, 10):
        cv2.line(gray, (i, 0), (i, size), 255, 1)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dark_rgb(size: int = 200) -> np.ndarray:
    gray = np.full((size, size), 15, dtype = np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope = "module")
def demo_module():
    from demo import app

    return app


def test_build_app_returns_blocks(demo_module):
    blocks = demo_module.build_app()
    assert blocks is not None


def test_analyze_none_returns_placeholder(demo_module):
    result = demo_module.analyze(None, "dataset_cleaner", False)
    assert result[0] == "No image"
    assert result[-1] == "{}"


def test_analyze_sharp_with_dataset_cleaner(demo_module):
    img = _sharp_rgb()
    verdict, summary, metrics, issues, preview, json_text = demo_module.analyze(
        img,
        "dataset_cleaner",
        False,
    )
    assert "PASS" in verdict
    assert len(summary) > 0
    assert "hf_energy_ratio" in metrics
    assert preview is not None

    payload = json.loads(json_text)
    assert payload["passed"] is True
    assert payload["preset"] == "dataset_cleaner"
    assert "hf_energy_ratio" in payload["metrics"]


def test_analyze_dark_fails_with_dataset_cleaner(demo_module):
    img = _dark_rgb()
    verdict, summary, _, issues, _, _ = demo_module.analyze(img, "dataset_cleaner", False)
    assert "FAIL" in verdict
    assert len(summary) > 10
    assert len(issues) > 0