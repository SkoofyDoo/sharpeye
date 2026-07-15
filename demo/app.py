"""SharpEye Gradio demo — portfolio showcase."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from sharpeye.config import list_presets
from sharpeye.ingest import collect_images
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

_EMPTY_BATCH_STATE: dict = {"order": [], "cache": {}, "selected_idx": None}


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

    json_text = json.dumps(report.to_dict(), indent=2)
    preview = _edge_overlay(bgr) if show_edges else image

    return verdict, summary, metrics_text, issues_text, preview, json_text


def _empty_zip_outputs() -> tuple:
    return (
        "_Upload a .zip archive and click Scan._",
        [],
        "{}",
        _EMPTY_BATCH_STATE,
        None,
        "Select a row to inspect a frame.",
        "",
        "",
        "",
        "{}",
    )


def scan_zip(
    zip_file: str | None,
    preset: str,
    show_edges: bool,
) -> tuple:
    if not zip_file:
        return _empty_zip_outputs()

    archive_path = Path(zip_file)
    if archive_path.suffix.lower() != ".zip":
        return (
            "Only .zip archives are supported.",
            [],
            "{}",
            _EMPTY_BATCH_STATE,
            None,
            "",
            "",
            "",
            "",
            "{}",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="sharpeye_demo_") as tmp:
            staging = Path(tmp) / archive_path.name
            staging.write_bytes(archive_path.read_bytes())

            with collect_images(staging) as paths:
                pipe = _get_pipeline(preset)
                batch = pipe.evaluate_batch(paths)

                entries: list[tuple[str, object, object, str, bool]] = []
                cache: dict[str, dict] = {}
                for path, frame in zip(paths, batch.frames, strict=True):
                    bgr = cv2.imread(str(path))
                    rgb = (
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if bgr is not None
                        else None
                    )
                    entries.append(
                        (
                            path.name,
                            frame,
                            rgb,
                            frame.human_summary,
                            frame.passed,
                        )
                    )

                entries.sort(key=lambda e: e[4])

                rows: list[list[str]] = []
                order: list[str] = []
                for name, frame, rgb, summary, passed in entries:
                    order.append(name)
                    cache[name] = {
                        "report": frame.to_dict(),
                        "rgb": rgb,
                    }
                    rows.append(
                        [
                            name,
                            "PASS" if passed else "FAIL",
                            frame.label,
                            summary[:120],
                        ]
                    )

                summary = (
                    f"**{batch.total}** frames — "
                    f"**{batch.passed_count}** passed, "
                    f"**{batch.failed_count}** failed "
                    f"(preset: `{preset}`)"
                )
                items = []
                for name in order:
                    item = dict(cache[name]["report"])
                    item["filename"] = name
                    items.append(item)
                payload = {**batch.to_dict(), "items": items}
                json_text = json.dumps(payload, indent=2)
                state = {"order": order, "cache": cache, "selected_idx": None}

                return (
                    summary,
                    rows,
                    json_text,
                    state,
                    None,
                    "_Click a row in the table to inspect a frame._",
                    "",
                    "",
                    "",
                    "{}",
                )

    except Exception as exc:
        return (
            f"Scan failed: {exc}",
            [],
            "{}",
            _EMPTY_BATCH_STATE,
            None,
            "",
            "",
            "",
            "",
            "{}",
        )


def _render_frame(state: dict, row_idx: int | None, show_edges: bool) -> tuple:
    if not state or not state.get("order") or row_idx is None:
        return None, "", "", "", "", "{}"

    order = state["order"]
    if row_idx < 0 or row_idx >= len(order):
        return None, "", "", "", "", "{}"

    name = order[row_idx]
    entry = state["cache"].get(name)
    if not entry:
        return None, "", "", "", "", "{}"

    report = entry["report"]
    rgb = entry["rgb"]
    if rgb is None:
        return None, "", "", "", "", json.dumps(report, indent=2)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    preview = _edge_overlay(bgr) if show_edges else rgb

    verdict = f"{'PASS' if report['passed'] else 'FAIL'} — {report['label'].upper()}"
    metrics_text = _format_metrics(report["metrics"])
    issues_lines = [
        f"- **{i['severity']}** {i['message']} → _{i['suggestion']}_"
        for i in report.get("issues", [])
    ]
    issues_text = (
        "\n".join(issues_lines) if issues_lines else "_No issues detected._"
    )

    return (
        preview,
        verdict,
        report.get("human_summary", ""),
        metrics_text,
        issues_text,
        json.dumps(report, indent=2),
    )


def show_frame_detail(
    evt: gr.SelectData,
    state: dict,
    show_edges: bool,
) -> tuple:
    row_idx = evt.index[0] if evt.index is not None else None
    if state and row_idx is not None:
        state = {**state, "selected_idx": row_idx}
    return (*_render_frame(state, row_idx, show_edges), state)


def refresh_frame_detail(state: dict, show_edges: bool) -> tuple:
    row_idx = state.get("selected_idx") if state else None
    return _render_frame(state, row_idx, show_edges)


def build_app() -> gr.Blocks:
    presets = list_presets()

    with gr.Blocks(title="SharpEye — Image Quality Control") as demo:
        gr.Markdown(
            "# SharpEye\n"
            "Eagle-eyed image quality control — human-readable verdicts "
            "for people and agents.\n\n"
            "**Presets:** `dataset_cleaner` — ML dataset QC; "
            "`telemedicine` — patient-friendly reshoot hints; "
            "`default` — general baseline."
        )

        with gr.Tabs():
            with gr.Tab("Single image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(label="Upload image", type="numpy")
                        preset_dd = gr.Dropdown(
                            choices=presets,
                            value="dataset_cleaner",
                            label="Preset",
                        )
                        show_edges = gr.Checkbox(
                            label="Show edge overlay",
                            value=False,
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=1):
                        verdict_out = gr.Textbox(label="Verdict", interactive=False)
                        summary_out = gr.Textbox(
                            label="Human summary",
                            lines=3,
                            interactive=False,
                        )
                        metrics_out = gr.Markdown(label="Metrics")
                        issues_out = gr.Markdown(label="Issues")
                        image_out = gr.Image(label="Analysis view")
                        json_out = gr.Textbox(
                            label="JSON report",
                            lines=12,
                            interactive=False,
                        )

                single_inputs = [image_in, preset_dd, show_edges]
                single_outputs = [
                    verdict_out,
                    summary_out,
                    metrics_out,
                    issues_out,
                    image_out,
                    json_out,
                ]

                analyze_btn.click(analyze, inputs=single_inputs, outputs=single_outputs)
                image_in.change(analyze, inputs=single_inputs, outputs=single_outputs)
                preset_dd.change(analyze, inputs=single_inputs, outputs=single_outputs)
                show_edges.change(analyze, inputs=single_inputs, outputs=single_outputs)

            with gr.Tab("Dataset (ZIP)"):
                batch_state = gr.State(_EMPTY_BATCH_STATE)

                with gr.Row():
                    with gr.Column(scale=1):
                        zip_in = gr.File(
                            label="Upload .zip archive",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        zip_preset_dd = gr.Dropdown(
                            choices=presets,
                            value="dataset_cleaner",
                            label="Preset",
                        )
                        zip_show_edges = gr.Checkbox(
                            label="Show edge overlay on selected frame",
                            value=False,
                        )
                        scan_btn = gr.Button("Scan archive", variant="primary")

                    with gr.Column(scale=1):
                        batch_summary = gr.Markdown("_No scan yet._")
                        results_df = gr.Dataframe(
                            headers=["File", "Verdict", "Label", "Summary"],
                            datatype=["str", "str", "str", "str"],
                            label="Results (failures first)",
                            interactive=False,
                        )
                        batch_json = gr.Textbox(
                            label="Batch JSON",
                            lines=10,
                            interactive=False,
                        )

                gr.Markdown("### Frame detail — click a row above")
                with gr.Row():
                    with gr.Column(scale=1):
                        frame_preview = gr.Image(label="Selected frame")
                    with gr.Column(scale=1):
                        frame_verdict = gr.Textbox(label="Verdict", interactive=False)
                        frame_summary = gr.Textbox(
                            label="Human summary",
                            lines=2,
                            interactive=False,
                        )
                        frame_metrics = gr.Markdown(label="Metrics")
                        frame_issues = gr.Markdown(label="Issues")
                        frame_json = gr.Textbox(
                            label="Frame JSON",
                            lines=8,
                            interactive=False,
                        )

                zip_outputs = [
                    batch_summary,
                    results_df,
                    batch_json,
                    batch_state,
                    frame_preview,
                    frame_verdict,
                    frame_summary,
                    frame_metrics,
                    frame_issues,
                    frame_json,
                ]

                scan_btn.click(
                    scan_zip,
                    inputs=[zip_in, zip_preset_dd, zip_show_edges],
                    outputs=zip_outputs,
                )
                results_df.select(
                    show_frame_detail,
                    inputs=[batch_state, zip_show_edges],
                    outputs=[
                        frame_preview,
                        frame_verdict,
                        frame_summary,
                        frame_metrics,
                        frame_issues,
                        frame_json,
                        batch_state,
                    ],
                )
                zip_show_edges.change(
                    refresh_frame_detail,
                    inputs=[batch_state, zip_show_edges],
                    outputs=[
                        frame_preview,
                        frame_verdict,
                        frame_summary,
                        frame_metrics,
                        frame_issues,
                        frame_json,
                    ],
                )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)