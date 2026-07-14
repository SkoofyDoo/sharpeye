"""Image preprocessing — resize, grayscale, ROI, shared ctx cache."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sharpeye.config import PreprocessConfig
from sharpeye.exceptions import InvalidImageError


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """ Loading Image """ 
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise InvalidImageError("Empty image array.")
        return source
    
    path = Path(source)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise InvalidImageError(f"Cannot read image: {path}")
    return img

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes(multipart) to BGR ndarray"""
    
    if not data:
        raise InvalidImageError("Empty image upload.")
    arr = np.frombuffer(data, dtype = np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Cannot decode image bytes.")
    return img

def _resize_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """ Resizing of Image to get better performance """
    # Image has 3 shapes (height, width, colorchannels)
    # for this operation only two needed
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

def _to_gray(img: np.ndarray) -> np.ndarray:
    """ Image to Grayscale """
    if img.ndim == 2:
        return img
    
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    raise InvalidImageError(f"Unsorpported image shape: {img.shape}")

def _extract_roi(gray: np.ndarray, ratio: float) -> np.ndarray:
    """ Extracting Region Of Interest """
    h, w = gray.shape[:2]
    
    reducedHeight = max(1, int(h * ratio))
    reducedWidth = max(1, int(w * ratio))

    y0 = (h - reducedHeight) // 2
    x0 = (w - reducedWidth) // 2

    return gray[y0 : y0 + reducedHeight, x0 : x0 + reducedWidth]

def preprocess_frame(
    source: str | Path | np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray, dict]:
    bgr = load_image(source)
    resized = _resize_max_width(bgr, config.max_width)
    gray = _to_gray(resized)
    roi_gray = _extract_roi(gray, config.roi_ratio)

    ctx: dict = {
        "bgr": resized,
        "gray": gray,
        "roi_gray": roi_gray,
        "roi_ratio": config.roi_ratio,
    }

    return gray, ctx