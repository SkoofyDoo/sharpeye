"""Milestone 3.2 — tool schema tests."""

from sharpeye.integrations.tool_schema import build_tool_schema, build_tools


def test_build_tool_schema_structure():
    schema = build_tool_schema()
    assert schema["name"] == "sharpeye_check_image"
    assert "parameters" in schema
    assert "image_base64" in schema["parameters"]["properties"]
    assert "preset" in schema["parameters"]["properties"]
    assert "use_case" in schema["parameters"]["properties"]


def test_build_tool_schema_includes_presets():
    schema = build_tool_schema()
    presets = schema["parameters"]["properties"]["preset"]["enum"]
    assert "default" in presets
    assert "telemedicine" in presets
    assert "dataset_cleaner" in presets


def test_build_tool_schema_api_endpoint():
    schema = build_tool_schema()
    assert schema["api"]["path"] == "/v1/check/b64"
    assert schema["api"]["method"] == "POST"


def test_build_tools_has_three_tools():
    tools = build_tools()
    names = {t["name"] for t in tools}
    assert names == {
        "sharpeye_check_image",
        "sharpeye_check_archive",
        "sharpeye_list_presets",
    }


def test_build_tools_archive_uses_b64_endpoint():
    tools = build_tools()
    archive = next(t for t in tools if t["name"] == "sharpeye_check_archive")
    assert archive["api"]["path"] == "/v1/batch/archive/b64"
    assert "archive_base64" in archive["parameters"]["properties"]