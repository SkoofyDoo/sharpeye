# SharpEye

[![Pipeline](https://github.com/SkoofyDoo/sharpeye/actions/workflows/pipeline.yml/badge.svg)](https://github.com/SkoofyDoo/sharpeye/actions/workflows/pipeline.yml)

**An eagle-eyed image quality control library.**

SharpEye evaluates image quality and returns human-readable verdicts with actionable suggestions — not just scores. YAML presets let you tune gates per use case: ML dataset cleaning, telemedicine photos, or general QC.

## Why SharpEye (vs raw metrics / piq)

| | piq / scores only | SharpEye |
|---|---|---|
| Output | `0.42` | `FAIL — Photo looks blurry. Suggestion: Hold the camera steady.` |
| Configuration | Code constants | YAML presets (`dataset_cleaner`, `telemedicine`) |
| Batch + CI | DIY | `sharpeye clean ./data --report csv` |
| Agents | DIY wrapper | `GET /v1/schema/tool` + structured `FrameReport` |
| Dataset-specific | Generic NR-IQA | `hf_energy_ratio` for JPEG/compression artifacts |

**Primary use case (Phase 1):** filter bad frames before labeling or training — blur, exposure outliers, heavy JPEG compression.

## Quick Start

```bash
pip install -e ".[cli]"
```

### Python API

```python
from sharpeye import Pipeline

pipe = Pipeline.from_preset("dataset_cleaner")
report = pipe.evaluate_frame("photo.jpg")

print(report.passed)
print(report.human_summary)
print(report.metrics)
print(report.to_dict())
```

### CLI — batch dataset cleaning

```bash
pip install -e ".[cli]"
sharpeye clean ./my_dataset --preset dataset_cleaner --report csv -o report.csv
```

### REST API

```bash
pip install -e ".[api]"
uvicorn api.main:app --reload
```

### Gradio Demo

```bash
pip install -e ".[demo]"
python demo/app.py
```

Two tabs: **Single image** (verdict + edge overlay) and **Dataset (ZIP)** (batch table, click a row for per-frame metrics and preview). Default preset is `dataset_cleaner`.

### Agent tool calling

Agents should use JSON endpoints (no multipart). Fetch tool definitions once:

```bash
curl http://127.0.0.1:8000/v1/schema/tools
```

Example — check a ZIP dataset as base64 with automatic preset from use case:

```python
import base64
import httpx

archive_b64 = base64.b64encode(open("dataset.zip", "rb").read()).decode()
r = httpx.post(
    "http://127.0.0.1:8000/v1/batch/archive/b64",
    json={"archive_base64": archive_b64, "use_case": "ml_dataset"},
)
print(r.json()["failed_count"], r.json()["items"][0]["human_summary"])
```

`use_case` maps to presets: `ml_dataset` → `dataset_cleaner`, `patient_photo` → `telemedicine`, `general` → `default`. Or pass `preset` directly.

## Presets

| Preset | Audience | Key metrics | Typical use |
|--------|----------|-------------|-------------|
| `dataset_cleaner` | ML engineers | + `hf_energy_ratio` | Dataset QC, CI pipelines |
| `telemedicine` | Patients / clinicians | blur, brightness, contrast | Reshoot hints |
| `default` | General | 6 core metrics | Baseline QC |

## Development

```bash
pip install -e ".[dev,cli,api,demo]"
pytest -q
ruff check src tests demo
```

## License

MIT — see [LICENSE](LICENSE).
