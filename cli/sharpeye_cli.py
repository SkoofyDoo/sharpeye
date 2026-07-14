"""SharpEye CLI - dataset cleaning and batch reports."""

import csv
import json
from pathlib import Path
from typing import Annotated

import typer

from sharpeye.pipeline import Pipeline

app = typer.Typer(
    name = "sharpeye",
    help = "SharpEye image quality control CLI",
    no_args_is_help = True,
)

_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


def _collect_images(dataset: Path) -> list[Path]:
    if not dataset.exists():
        raise typer.BadParameter(f"Dataset path not found: {dataset}")
    if dataset.is_file():
        if dataset.suffix.lower() not in _IMAGE_EXTS:
            raise typer.BadParameter(f"Not an image file: {dataset}")
        return [dataset]
    files = sorted(
        p for p in dataset.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not files:
        raise typer.BadParameter(f"No images found under: {dataset}")
    return files


def _write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.command("version")
def version() -> None:
    """Show SharpEye version."""
    typer.echo("sharpeye 0.1.0")


@app.command("clean")
def clean(
    dataset: Annotated[str, typer.Argument(help = "Folder or single image path")],
    preset: Annotated[str, typer.Option("--preset", "-p")] = "dataset_cleaner",
    report: Annotated[str, typer.Option("--report", "-r", help = "csv or json")] = "csv",
    output: Annotated[
        str | None, typer.Option("--output", "-o", help = "Report file path"),
    ] = None,
) -> None:
    """Scan dataset, run batch QC, write report."""
    paths = _collect_images(Path(dataset))
    pipe = Pipeline.from_preset(preset)
    batch = pipe.evaluate_batch(paths)

    rows: list[dict] = []
    for path, frame in zip(paths, batch.frames, strict = True):
        row = {
            "path": str(path),
            "passed": frame.passed,
            "label": frame.label,
            "human_summary": frame.human_summary,
            "composite_score": frame.composite_score,
        }
        for k, v in frame.metrics.items():
            row[f"metric_{k}"] = round(v, 4)
        rows.append(row)

    out_path = Path(output) if output is not None else Path(
        f"sharpeye_report.{'json' if report == 'json' else 'csv'}"
    )

    if report == "json":
        payload = {
            "preset": batch.preset,
            "total": batch.total,
            "passed_count": batch.passed_count,
            "failed_count": batch.failed_count,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    elif report == "csv":
        _write_csv(rows, out_path)
    else:
        raise typer.BadParameter("report must be 'csv' or 'json'")

    typer.echo(
        f"Done: {batch.passed_count}/{batch.total} passed. Report: {out_path}"
    )


def run_app() -> None:
    app()


if __name__ == "__main__":
    run_app()