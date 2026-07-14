"""Milestone 3.2 — REST API tests."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
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