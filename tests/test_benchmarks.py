"""Benchmark tests against ground_truth.json."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sharpeye import Pipeline

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"
PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"


def _sharp_image(size: int = 200) -> np.ndarray:
    img = np.full((size, size), 80, dtype = np.uint8)
    for i in range(0, size, 10):
        cv2.line(img, (i, 0), (i, size), 255, 1)
    return img


def _blur_image(size: int = 200, ksize: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size), (ksize, ksize), 0)


def _dark_image(size: int = 200) -> np.ndarray:
    return np.full((size, size), 15, dtype = np.uint8)


_IMAGE_BUILDERS = {
    "sharp": _sharp_image,
    "blur": _blur_image,
    "dark": _dark_image,
}


@pytest.fixture(scope = "module")
def ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding = "utf-8"))


@pytest.fixture(scope = "module")
def pipe(ground_truth) -> Pipeline:
    return Pipeline.from_preset(ground_truth["preset"], presets_dir = PRESETS_DIR)


@pytest.mark.parametrize("case_id", ["sharp", "blur", "dark"])
def test_benchmark_case(case_id: str, ground_truth: dict, pipe: Pipeline) -> None:
    case = next(c for c in ground_truth["cases"] if c["id"] == case_id)
    expected = case["expected"]
    report = pipe.evaluate_frame(_IMAGE_BUILDERS[case_id]())

    assert report.passed == expected["passed"]

    blur_hit = any(
        i.metric == "laplacian_variance" and i.severity == "catastrophic"
        for i in report.issues
    )
    dark_hit = any(
        i.metric == "brightness_mean" and i.severity == "catastrophic"
        for i in report.issues
    )

    assert blur_hit == expected["blur"]
    assert dark_hit == expected["dark"]