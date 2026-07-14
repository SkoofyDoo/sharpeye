"""Milestone 3.3 — CLI tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from cli.sharpeye_cli import _collect_images, app

runner = CliRunner()


def _write_sharp_png(path: Path) -> None:
    img = np.full((120, 120), 80, dtype = np.uint8)
    for i in range(0, 120, 8):
        cv2.line(img, (i, 0), (i, 120), 255, 1)
    cv2.imwrite(str(path), img)


def _write_dark_png(path: Path) -> None:
    img = np.full((120, 120), 15, dtype = np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def fixture_images(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_sharp_png(dataset / "sharp.png")
    _write_dark_png(dataset / "dark.png")
    return dataset


def test_collect_images_folder(fixture_images: Path):
    paths = _collect_images(fixture_images)
    assert len(paths) == 2


def test_collect_images_single_file(fixture_images: Path):
    paths = _collect_images(fixture_images / "sharp.png")
    assert len(paths) == 1


def test_clean_csv_report(fixture_images: Path, tmp_path: Path):
    out = tmp_path / "report.csv"
    result = runner.invoke(
        app,
        [
            "clean",
            str(fixture_images),
            "--preset", "dataset_cleaner",
            "--report", "csv",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    with out.open(encoding = "utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "metric_hf_energy_ratio" in rows[0]


def test_clean_json_report(fixture_images: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "clean",
            str(fixture_images),
            "--preset", "default",
            "--report", "json",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding = "utf-8"))
    assert data["total"] == 2
    assert data["failed_count"] >= 1


def test_clean_missing_dataset():
    result = runner.invoke(app, ["clean", "nonexistent_folder_xyz"])
    assert result.exit_code != 0


def test_clean_single_image(fixture_images: Path, tmp_path: Path):
    image = fixture_images / "sharp.png"
    out = tmp_path / "one.csv"
    result = runner.invoke(
        app,
        ["clean", str(image), "--preset", "default", "-o", str(out)],
    )
    assert result.exit_code == 0
    with out.open(encoding = "utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1