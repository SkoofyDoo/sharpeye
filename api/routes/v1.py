"""REST API v1 Routes"""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from sharpeye.config import list_presets_catalog, resolve_preset
from sharpeye.exceptions import (
    ArchiveError,
    InvalidImageError,
    PresetNotFoundError,
    SharpEyeError,
)
from sharpeye.ingest import collect_images, is_archive
from sharpeye.integrations.tool_schema import build_tool_schema, build_tools
from sharpeye.pipeline import Pipeline
from sharpeye.preprocess import load_image_from_bytes

router = APIRouter(prefix="/v1")

_MAX_BATCH_FILES = 50


class CheckB64Request(BaseModel):
    image_base64: str
    preset: str | None = None
    use_case: str | None = None


class ArchiveB64Request(BaseModel):
    archive_base64: str
    preset: str | None = None
    use_case: str | None = None


def _decode_base64(data: str) -> bytes:
    payload = data.strip()
    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail="Invalid base64 payload.") from e


def _run_archive_batch(data: bytes, preset_name: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="sharpeye_api_") as tmp:
        archive_path = Path(tmp) / "upload.zip"
        archive_path.write_bytes(data)

        with collect_images(archive_path) as paths:
            if len(paths) > _MAX_BATCH_FILES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Maximum {_MAX_BATCH_FILES} images per archive batch.",
                )
            pipe = Pipeline.from_preset(preset_name)
            batch = pipe.evaluate_batch(paths)

            items = []
            for path, frame in zip(paths, batch.frames, strict=True):
                item = frame.to_dict()
                item["filename"] = path.name
                items.append(item)

            return {**batch.to_dict(), "items": items}


@router.get("/presets")
def get_presets() -> dict:
    catalog = list_presets_catalog()
    names = [str(entry["name"]) for entry in catalog]
    return {"presets": names, "count": len(names), "catalog": catalog}


@router.get("/schema/tool")
def get_tool_schema() -> dict:
    return build_tool_schema()


@router.get("/schema/tools")
def get_tools_schema() -> dict:
    return {"tools": build_tools(), "count": len(build_tools())}


@router.post("/check")
async def check_image(
    file: UploadFile = File(...),
    preset: str = Form(default="default"),
) -> dict:
    try:
        data = await file.read()
        img = load_image_from_bytes(data)
        pipe = Pipeline.from_preset(preset)
        report = pipe.evaluate_frame(img)
        return report.to_dict()

    except PresetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/check/b64")
def check_image_b64(body: CheckB64Request) -> dict:
    try:
        preset_name = resolve_preset(preset=body.preset, use_case=body.use_case)
        data = _decode_base64(body.image_base64)
        img = load_image_from_bytes(data)
        pipe = Pipeline.from_preset(preset_name)
        report = pipe.evaluate_frame(img)
        return report.to_dict()

    except HTTPException:
        raise
    except PresetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/batch")
async def check_batch(
    files: list[UploadFile] = File(...),
    preset: str = Form(default="default"),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {_MAX_BATCH_FILES} files per batch.",
        )
    try:
        images = []
        for f in files:
            data = await f.read()
            images.append(load_image_from_bytes(data))
        pipe = Pipeline.from_preset(preset)
        batch = pipe.evaluate_batch(images)
        return batch.to_dict()

    except PresetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/batch/archive")
async def check_batch_archive(
    file: UploadFile = File(...),
    preset: str = Form(default="dataset_cleaner"),
) -> dict:
    filename = file.filename or "archive.zip"
    if not is_archive(Path(filename)):
        raise HTTPException(status_code=400, detail="Only .zip archives are supported.")

    try:
        data = await file.read()
        return _run_archive_batch(data, preset)

    except HTTPException:
        raise
    except PresetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ArchiveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/batch/archive/b64")
def check_batch_archive_b64(body: ArchiveB64Request) -> dict:
    try:
        preset_name = resolve_preset(preset=body.preset, use_case=body.use_case)
        data = _decode_base64(body.archive_base64)
        return _run_archive_batch(data, preset_name)

    except HTTPException:
        raise
    except PresetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ArchiveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e