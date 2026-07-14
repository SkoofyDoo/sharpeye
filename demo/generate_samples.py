"""Optional dev utility — synthetic images for local experiments.

Not used by the Gradio UI (upload-only). Replace with real photos later.
Run: python demo/generate_samples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SIZE = 320


def _sharp(size: int = SIZE) -> np.ndarray:
    gray = np.full((size, size), 80, dtype = np.uint8)
    for i in range(0, size, 10):
        cv2.line(gray, (i, 0), (i, size), 255, 1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _blur(size: int = SIZE) -> np.ndarray:
    return cv2.GaussianBlur(_sharp(size), (21, 21), 0)


def _dark(size: int = SIZE) -> np.ndarray:
    gray = np.full((size, size), 15, dtype = np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _noisy(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = _sharp(size).astype(np.int16)
    noise = rng.integers(-30, 30, size = (size, size, 3), dtype = np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _low_contrast(size: int = SIZE) -> np.ndarray:
    gray = np.full((size, size), 127, dtype = np.uint8)
    for i in range(0, size, 4):
        cv2.line(gray, (i, 0), (i, size), 133, 1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _compressed(size: int = SIZE, quality: int = 5) -> np.ndarray:
    """Heavy JPEG recompression — triggers hf_energy_ratio gate."""
    ok, buf = cv2.imencode(".jpg", _sharp(size), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return decoded


def main() -> None:
    SAMPLES_DIR.mkdir(parents = True, exist_ok = True)
    samples = {
        "sharp": _sharp(),
        "blur": _blur(),
        "dark": _dark(),
        "noisy": _noisy(),
        "low_contrast": _low_contrast(),
        "compressed": _compressed(),
    }
    for name, img in samples.items():
        path = SAMPLES_DIR / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()