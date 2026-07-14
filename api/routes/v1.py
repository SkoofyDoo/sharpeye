"""REST API v1 Routes"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from sharpeye.config import list_presets
from sharpeye.exceptions import InvalidImageError, PresetNotFoundError, SharpEyeError
from sharpeye.integrations.tool_schema import build_tool_schema
from sharpeye.pipeline import Pipeline
from sharpeye.preprocess import load_image_from_bytes

router = APIRouter(prefix = "/v1")

_MAX_BATCH_FILES = 50

@router.get("/presets")
def get_presets() -> dict:
    names = list_presets()
    return {"presets": names, "count": len(names)}

@router.get("/schema/tool")
def get_tool_schema() -> dict:
    return build_tool_schema()

@router.post("/check")
async def check_image(
    file: UploadFile = File(...),
    preset: str = Form(default = "default"),
) -> dict:
    try:
        data = await file.read()
        img = load_image_from_bytes(data)
        pipe = Pipeline.from_preset(preset)
        report = pipe.evaluate_frame(img)
        return report.to_dict()
    
    except PresetNotFoundError as e:
        raise HTTPException(status_code = 404, detail = str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code = 400, detail = str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code = 422, detail = str(e)) from e

@router.post("/batch")
async def check_batch(
    files: list[UploadFile] = File(...),
    preset: str = Form(default = "default"),
) -> dict:
    if not files:
        raise HTTPException(status_code = 400, detail = "No files uploaded.")
    
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code = 400,
            detail = f"Maximum {_MAX_BATCH_FILES} files per batch."
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
        raise HTTPException(status_code = 404, detail = str(e)) from e
    except InvalidImageError as e:
        raise HTTPException(status_code = 400, detail = str(e)) from e
    except SharpEyeError as e:
        raise HTTPException(status_code = 422, detail = str(e)) from e
