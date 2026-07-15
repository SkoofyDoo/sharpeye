"""Preset configuration — loaded from YAML, validated by Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from sharpeye.exceptions import PresetNotFoundError, PresetValidationError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRESETS_DIR = _PROJECT_ROOT / "presets"


class PreprocessConfig(BaseModel):
    max_width: int = 640
    roi_ratio: float = Field(default=0.65, ge=0.1, le=1.0)

class MetricsConfig(BaseModel):
    noise_kernel_size: int = Field(default = 7, ge = 3, le = 31)

class GateRule(BaseModel):
    metric: str
    min: float | None = None
    max: float | None = None
    severity: Literal["catastrophic", "soft"] = "catastrophic"
    message: str = ""
    suggestion: str = ""

class GateGroup(BaseModel):
    mode: Literal["any_of", "all_of"] = "any_of"
    rules: list[GateRule] = Field(default_factory=list)


class GatesConfig(BaseModel):
    rules: list[GateRule] = Field(default_factory=list)
    groups: list[GateGroup] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    good_percentile: float = Field(default=0.67, ge=0.0, le=1.0)
    medium_percentile: float = Field(default=0.33, ge=0.0, le=1.0)

    @field_validator("weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            return v
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Composite weights must sum to 1.0, got {total:.3f}")
        return v


class Preset(BaseModel):
    name: str
    description: str = ""
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    enabled_metrics: list[str] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory = MetricsConfig)


def _presets_dir() -> Path:
    return _PRESETS_DIR


def load_preset(name: str, presets_dir: Path | None = None) -> Preset:
    directory = presets_dir or _presets_dir()
    path = directory / f"{name}.yaml"

    if not path.exists():
        raise PresetNotFoundError(f"Preset not found: {path}")

    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("name", name)
        return Preset.model_validate(raw)
    except PresetValidationError:
        raise
    except Exception as e:
        raise PresetValidationError(f"Invalid preset '{name}': {e}") from e


def list_presets(presets_dir: Path | None = None) -> list[str]:
    directory = presets_dir or _presets_dir()
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


USE_CASE_PRESETS: dict[str, str] = {
    "ml_dataset": "dataset_cleaner",
    "dataset_qc": "dataset_cleaner",
    "telemedicine": "telemedicine",
    "patient_photo": "telemedicine",
    "general": "default",
}


def list_presets_catalog(presets_dir: Path | None = None) -> list[dict[str, object]]:
    directory = presets_dir or _presets_dir()
    catalog: list[dict[str, object]] = []
    for name in list_presets(directory):
        preset = load_preset(name, directory)
        use_cases = sorted(k for k, v in USE_CASE_PRESETS.items() if v == name)
        catalog.append(
            {
                "name": name,
                "description": preset.description,
                "use_cases": use_cases,
                "enabled_metrics": preset.enabled_metrics,
            }
        )
    return catalog


def resolve_preset(
    *,
    preset: str | None = None,
    use_case: str | None = None,
    presets_dir: Path | None = None,
) -> str:
    directory = presets_dir or _presets_dir()
    available = list_presets(directory)

    if preset:
        if preset not in available:
            raise PresetNotFoundError(f"Preset not found: {preset}")
        return preset

    if use_case:
        key = use_case.strip().lower()
        mapped = USE_CASE_PRESETS.get(key)
        if mapped is None:
            known = ", ".join(sorted(USE_CASE_PRESETS))
            raise PresetNotFoundError(
                f"Unknown use_case '{use_case}'. Known values: {known}"
            )
        return mapped

    return "default"