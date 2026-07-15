"""SharpEye CLI - dataset cleaning and batch reports."""

import csv
import json
from pathlib import Path
from typing import Annotated

import typer

from sharpeye.exceptions import ArchiveError, PresetNotFoundError, SharpEyeError
from sharpeye.ingest import collect_images
from sharpeye.pipeline import Pipeline

app = typer.Typer(
    name="sharpeye",
    help="SharpEye image quality control CLI",
    no_args_is_help=True,
)


def _write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.command("version")
def version() -> None:
    """Show SharpEye version."""
    typer.echo("sharpeye 0.1.0")


@app.command("clean")
def clean(
    dataset: Annotated[
        str,
        typer.Argument(help="Folder, .zip archive, or single image"),
    ],
    preset: Annotated[str, typer.Option("--preset", "-p")] = "dataset_cleaner",
    report: Annotated[
        str, typer.Option("--report", "-r", help="csv or json")
    ] = "csv",
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Report file path")
    ] = None,
) -> None:
    """Scan dataset, run batch QC, write report."""
    try:
        with collect_images(Path(dataset)) as paths:
            pipe = Pipeline.from_preset(preset)
            batch = pipe.evaluate_batch(paths)

            rows: list[dict] = []
            for path, frame in zip(paths, batch.frames, strict=True):
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
    except ArchiveError as e:
        raise typer.BadParameter(str(e)) from e
    except PresetNotFoundError as e:
        raise typer.BadParameter(str(e)) from e
    except SharpEyeError as e:
        raise typer.BadParameter(str(e)) from e

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
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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