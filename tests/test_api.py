"""Milestone 3.2 — REST API tests."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _png_bytes(gray: np.ndarray | None = None) -> bytes:
    if gray is None:
        gray = np.full((100, 100), 128, dtype = np.uint8)
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    return buf.tobytes()


def _sharp_png() -> bytes:
    img = np.full((100, 100), 80, dtype = np.uint8)
    for i in range(0, 100, 10):
        cv2.line(img, (i, 0), (i, 100), 255, 1)
    return _png_bytes(img)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_presets():
    r = client.get("/v1/presets")
    assert r.status_code == 200
    data = r.json()
    assert "default" in data["presets"]
    assert data["count"] >= 2


def test_get_tool_schema():
    r = client.get("/v1/schema/tool")
    assert r.status_code == 200
    assert r.json()["name"] == "sharpeye_check_image"


def test_get_tools_schema():
    r = client.get("/v1/schema/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    names = {t["name"] for t in body["tools"]}
    assert "sharpeye_check_archive" in names


def test_get_presets_includes_catalog():
    r = client.get("/v1/presets")
    assert r.status_code == 200
    body = r.json()
    assert "catalog" in body
    assert body["catalog"][0]["description"]


def test_check_sharp_image():
    r = client.post(
        "/v1/check",
        files = {"file": ("sharp.png", io.BytesIO(_sharp_png()), "image/png")},
        data = {"preset": "default"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is True
    assert body["label"] == "good"
    assert "laplacian_variance" in body["metrics"]


def test_check_dark_image_fails():
    dark = np.full((100, 100), 15, dtype = np.uint8)
    r = client.post(
        "/v1/check",
        files = {"file": ("dark.png", io.BytesIO(_png_bytes(dark)), "image/png")},
        data = {"preset": "default"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False
    assert body["label"] == "bad"


def test_check_invalid_preset():
    r = client.post(
        "/v1/check",
        files = {"file": ("x.png", io.BytesIO(_png_bytes()), "image/png")},
        data = {"preset": "nonexistent"},
    )
    assert r.status_code == 404


def test_check_invalid_image_bytes():
    r = client.post(
        "/v1/check",
        files = {"file": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
        data = {"preset": "default"},
    )
    assert r.status_code == 400


def test_batch_two_images():
    dark = np.full((100, 100), 15, dtype = np.uint8)
    r = client.post(
        "/v1/batch",
        files = [
            ("files", ("sharp.png", io.BytesIO(_sharp_png()), "image/png")),
            ("files", ("dark.png", io.BytesIO(_png_bytes(dark)), "image/png")),
        ],
        data = {"preset": "default"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["passed_count"] + body["failed_count"] == 2


def test_batch_empty_rejected():
    r = client.post("/v1/batch", data = {"preset": "default"})
    assert r.status_code == 422


def test_check_b64_sharp_image():
    payload = {
        "image_base64": base64.b64encode(_sharp_png()).decode("ascii"),
        "preset": "default",
    }
    r = client.post("/v1/check/b64", json=payload)
    assert r.status_code == 200
    assert r.json()["passed"] is True


def test_check_b64_use_case_resolves_preset():
    payload = {
        "image_base64": base64.b64encode(_sharp_png()).decode("ascii"),
        "use_case": "ml_dataset",
    }
    r = client.post("/v1/check/b64", json=payload)
    assert r.status_code == 200
    assert r.json()["preset"] == "dataset_cleaner"


def _make_zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_batch_archive_b64(tmp_path: Path):
    zbytes = _make_zip_bytes({
        "one.png": _png_bytes(),
        "two.png": _sharp_png(),
    })
    r = client.post(
        "/v1/batch/archive/b64",
        json={
            "archive_base64": base64.b64encode(zbytes).decode("ascii"),
            "use_case": "dataset_qc",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["preset"] == "dataset_cleaner"