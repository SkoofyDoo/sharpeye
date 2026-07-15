"""Preset catalog and use_case resolution tests."""

import pytest

from sharpeye.config import (
    list_presets_catalog,
    resolve_preset,
)
from sharpeye.exceptions import PresetNotFoundError


def test_list_presets_catalog_includes_descriptions():
    catalog = list_presets_catalog()
    names = {entry["name"] for entry in catalog}
    assert "dataset_cleaner" in names
    assert "telemedicine" in names
    for entry in catalog:
        assert entry["description"]
        assert entry["use_cases"]
        assert entry["enabled_metrics"]


def test_resolve_preset_by_name():
    assert resolve_preset(preset="dataset_cleaner") == "dataset_cleaner"


def test_resolve_preset_by_use_case():
    assert resolve_preset(use_case="ml_dataset") == "dataset_cleaner"
    assert resolve_preset(use_case="patient_photo") == "telemedicine"


def test_resolve_preset_unknown_use_case():
    with pytest.raises(PresetNotFoundError, match="Unknown use_case"):
        resolve_preset(use_case="underwater_robotics")


def test_resolve_preset_unknown_name():
    with pytest.raises(PresetNotFoundError):
        resolve_preset(preset="nonexistent")