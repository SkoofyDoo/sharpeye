"""JSON Schema for LLM tool calling — describes SharpEye REST API."""

from __future__ import annotations

from typing import Any

from sharpeye.config import USE_CASE_PRESETS, list_presets, list_presets_catalog


def build_tool_schema() -> dict[str, Any]:
    """Legacy single-tool schema (backward compatible)."""
    tools = build_tools()
    return tools[0]


def build_tools() -> list[dict[str, Any]]:
    presets = list_presets()
    catalog = list_presets_catalog()
    preset_guide = "; ".join(f"{c['name']}: {c['description']}" for c in catalog)
    use_cases = sorted(USE_CASE_PRESETS)

    return [
        {
            "name": "sharpeye_check_image",
            "description": (
                "Analyze a single image for quality. Returns verdict, metrics, "
                "issues, and human-readable suggestions. Images are not stored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image (jpg/png/webp).",
                    },
                    "preset": {
                        "type": "string",
                        "description": (
                            f"Quality preset. Options: {preset_guide}"
                        ),
                        "enum": presets,
                    },
                    "use_case": {
                        "type": "string",
                        "description": (
                            "Alternative to preset — maps to a recommended preset "
                            f"({', '.join(use_cases)})."
                        ),
                        "enum": use_cases,
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
                    "issues": {"type": "array"},
                },
            },
            "api": {"method": "POST", "path": "/v1/check/b64"},
        },
        {
            "name": "sharpeye_check_archive",
            "description": (
                "Scan all images inside a ZIP archive. Returns batch summary "
                "plus per-frame verdicts with filenames. Archives are not stored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "archive_base64": {
                        "type": "string",
                        "description": "Base64-encoded .zip archive containing images.",
                    },
                    "preset": {
                        "type": "string",
                        "description": (
                            f"Quality preset. Options: {preset_guide}"
                        ),
                        "enum": presets,
                    },
                    "use_case": {
                        "type": "string",
                        "description": (
                            "Alternative to preset — maps to a recommended preset "
                            f"({', '.join(use_cases)})."
                        ),
                        "enum": use_cases,
                    },
                },
                "required": ["archive_base64"],
            },
            "returns": {
                "type": "object",
                "description": "BatchReport with items[] including filename per frame.",
            },
            "api": {"method": "POST", "path": "/v1/batch/archive/b64"},
        },
        {
            "name": "sharpeye_list_presets",
            "description": (
                "List available quality presets with descriptions and use-case hints."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
            "returns": {
                "type": "object",
                "description": "Catalog of presets with use_cases and enabled_metrics.",
            },
            "api": {"method": "GET", "path": "/v1/presets"},
        },
    ]