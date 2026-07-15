"""Milestone 6.1 — archive ingest tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from sharpeye.exceptions import ArchiveError
from sharpeye.ingest import collect_images, is_archive

client = TestClient(app)


def _png_bytes() -> bytes:
    img = np.full((80, 80), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_zip(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    zpath = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return zpath


def test_is_archive():
    assert is_archive(Path("data.zip")) is True
    assert is_archive(Path("folder")) is False


def test_collect_images_from_zip(tmp_path: Path):
    zpath = _make_zip(tmp_path, {
        "a/sharp.png": _png_bytes(),
        "b/dark.png": _png_bytes(),
    })
    with collect_images(zpath) as paths:
        assert len(paths) == 2


def test_zip_slip_rejected(tmp_path: Path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escape.png", _png_bytes())

    with pytest.raises(ArchiveError, match="Unsafe|Zip-slip"):
        with collect_images(zpath):
            pass


def test_empty_zip_raises(tmp_path: Path):
    zpath = _make_zip(tmp_path, {"readme.txt": b"no images"})
    with pytest.raises(ArchiveError, match="No images"):
        with collect_images(zpath):
            pass


def test_api_batch_archive(tmp_path: Path):
    zpath = _make_zip(tmp_path, {"one.png": _png_bytes(), "two.png": _png_bytes()})
    with zpath.open("rb") as f:
        r = client.post(
            "/v1/batch/archive",
            files={"file": ("dataset.zip", f, "application/zip")},
            data={"preset": "dataset_cleaner"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert "filename" in body["items"][0]