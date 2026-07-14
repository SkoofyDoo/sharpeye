"""Milestone 3.2 — tool schema tests."""

from sharpeye.integrations.tool_schema import build_tool_schema


def test_build_tool_schema_structure():
    schema = build_tool_schema()
    assert schema["name"] == "sharpeye_check_image"
    assert "parameters" in schema
    assert "image_base64" in schema["parameters"]["properties"]
    assert "preset" in schema["parameters"]["properties"]


def test_build_tool_schema_includes_presets():
    schema = build_tool_schema()
    presets = schema["parameters"]["properties"]["preset"]["enum"]
    assert "default" in presets
    assert "telemedicine" in presets


def test_build_tool_schema_api_endpoints():
    schema = build_tool_schema()
    endpoints = schema["api"]["endpoints"]
    assert endpoints["presets"]["path"] == "/v1/presets"
    assert endpoints["check"]["method"] == "POST"