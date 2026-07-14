"""JSON Schema for LLM tool calling - describes SharpEye REST API"""

from __future__ import annotations

from typing import Any

from sharpeye.config import list_presets


def build_tool_schema() -> dict[str, Any]:
    presets = list_presets()
    return {
        "name": "sharpeye_check_image",
        "description": (
            "Analyze image quality using SharpEye. Returns human-readable verdict,"
            "metrics, and actionable suggestions. Images are not stored"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "description": "Quality preset name (domain-specific thresholds)",
                    "enum": presets,
                    "default": "default",
                },
                "image_base64": {
                    "type": "string",
                    "description": "Base64-encoded image (jpg/png/webp)."
                },

            },
            "required": ["image_base64"],
        },
        "returns": {
            "type": "object",
            "description": "FrameReport: passed, label, metrics, issues, human_summary.",
            "properties": {
                "passed": {"type": "boolean"},
                "label": {"type": "string", "enum": ["good", "medium", "bad"]},
                "human_summary": {"type": "string"}, 
                "metrics": {"type": "object"},
                "issues": {"type": "array"}
            },
        },
        "api": {
            "base_url": "/v1",
            "endpoints": {
                "check": {"method": "POST", "path": "/v1/check"},
                "batch": {"method": "POST", "path": "/v1/batch"},
                "presets": {"method": "GET", "path": "/v1/presets"},
            },
        },
    }
