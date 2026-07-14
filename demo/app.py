"""SharpEye Gradio demo — portfolio showcase."""

from __future__ import annotations

import json

import cv2
import gradio as gr
import numpy as np

from sharpeye.config import list_presets
from sharpeye.pipeline import Pipeline

_PRESET_CACHE: dict[str, Pipeline] = {}

_METRIC_CEIL: dict[str, float] = {
    "laplacian_variance": 500.0,
    "tenengrad": 5000.0,
    "contrast_std": 80.0,
    "brightness_mean": 255.0,
    "noise_std": 30.0,
    "edge_laplacian_p90": 200.0,
    "hf_energy_ratio": 0.5,
}


def _get_pipeline(preset: str) -> Pipeline:
    if preset not in _PRESET_CACHE:
        _PRESET_CACHE[preset] = Pipeline.from_preset(preset)
    return _PRESET_CACHE[preset]


def _metric_bar(name: str, value: float) -> str:
    ceiling = _METRIC_CEIL.get(name, max(abs(value), 1.0))
    pct = min(100, int(100 * abs(value) / ceiling))
    filled = pct // 5
    bar = "█" * filled + "░" * (20 - filled)
    return f"**{name}** &nbsp; `{value:.2f}` &nbsp; `{bar}` {pct}%"


def _format_metrics(metrics: dict[str, float]) -> str:
    lines = [_metric_bar(name, value) for name, value in sorted(metrics.items())]
    return "\n\n".join(lines)


def _edge_overlay(bgr: np.ndarray) -> np.ndarray:
    """Canny edges on ROI — 'what the algorithm sees'."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ratio = 0.65
    rh = max(1, int(h * ratio))
    rw = max(1, int(w * ratio))
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    roi = gray[y0 : y0 + rh, x0 : x0 + rw]
    median = float(np.median(roi))
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(roi, lower, upper)
    overlay = bgr.copy()
    roi_color = overlay[y0 : y0 + rh, x0 : x0 + rw]
    roi_color[edges > 0] = [0, 255, 0]
    overlay[y0 : y0 + rh, x0 : x0 + rw] = roi_color
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def analyze(
    image: np.ndarray | None,
    preset: str,
    show_edges: bool,
) -> tuple[str, str, str, str, np.ndarray | None, str]:
    if image is None:
        return "No image", "", "", "_Upload an image to analyze._", None, "{}"

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pipe = _get_pipeline(preset)
    report = pipe.evaluate_frame(bgr)

    verdict = f"{'PASS' if report.passed else 'FAIL'} — {report.label.upper()}"
    summary = report.human_summary
    metrics_text = _format_metrics(report.metrics)

    issues_lines = [
        f"- **{issue.severity}** {issue.message} → _{issue.suggestion}_"
        for issue in report.issues
    ]
    issues_text = "\n".join(issues_lines) if issues_lines else "_No issues detected._"

    json_text = json.dumps(report.to_dict(), indent = 2)
    preview = _edge_overlay(bgr) if show_edges else image

    return verdict, summary, metrics_text, issues_text, preview, json_text


def build_app() -> gr.Blocks:
    presets = list_presets()

    with gr.Blocks(title = "SharpEye — Image Quality Control") as demo:
        gr.Markdown(
            "# SharpEye\n"
            "Eagle-eyed image quality control — human-readable verdicts "
            "for people and agents.\n\n"
            "**Presets:** `dataset_cleaner` — ML dataset QC (blur, exposure, "
            "JPEG artifacts via `hf_energy_ratio`); "
            "`telemedicine` — patient-friendly reshoot hints."
        )

        with gr.Row():
            with gr.Column(scale = 1):
                image_in = gr.Image(label = "Upload image", type = "numpy")
                preset_dd = gr.Dropdown(
                    choices = presets,
                    value = "dataset_cleaner",
                    label = "Preset",
                )
                show_edges = gr.Checkbox(label = "Show edge overlay", value = False)
                analyze_btn = gr.Button("Analyze", variant = "primary")

            with gr.Column(scale = 1):
                verdict_out = gr.Textbox(label = "Verdict", interactive = False)
                summary_out = gr.Textbox(
                    label = "Human summary",
                    lines = 3,
                    interactive = False,
                )
                metrics_out = gr.Markdown(label = "Metrics")
                issues_out = gr.Markdown(label = "Issues")
                image_out = gr.Image(label = "Analysis view")
                json_out = gr.Textbox(
                    label = "JSON report",
                    lines = 12,
                    interactive = False,
                    buttons = ["copy"],
                )

        inputs = [image_in, preset_dd, show_edges]
        outputs = [
            verdict_out,
            summary_out,
            metrics_out,
            issues_out,
            image_out,
            json_out,
        ]

        analyze_btn.click(analyze, inputs = inputs, outputs = outputs)
        image_in.change(analyze, inputs = inputs, outputs = outputs)
        preset_dd.change(analyze, inputs = inputs, outputs = outputs)
        show_edges.change(analyze, inputs = inputs, outputs = outputs)

    return demo


if __name__ == "__main__":
    build_app().launch(server_name = "127.0.0.1", server_port = 7860)