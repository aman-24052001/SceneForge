"""
Orchestrator Module
Responsibilities:
- Glue fetcher -> validator -> colmap_runner -> engine -> viewer end-to-end
- Expose a CLI via Typer: `sceneforge run --images <folder> --out <folder>`
- Surface clear errors at whichever stage fails, without partial silent runs
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer

from sceneforge import colmap_runner, engine, fetcher, validator, viewer

app = typer.Typer(help="SceneForge: local images -> navigable 3D Gaussian Splat scene.")


@dataclass
class PipelineResult:
    html_path: Path
    ply_path: Path
    num_images: int


def run_pipeline(
    images_dir: str | Path,
    output_dir: str | Path,
    iterations: int = 1000,
    use_gpu: bool = False,
    opensplat_binary: str | None = None,
) -> PipelineResult:
    """
    Run the full SceneForge pipeline end-to-end.

    Raises whichever stage-specific exception occurs (FetchError,
    ColmapStageError, OpenSplatRunError, etc.) -- the caller decides
    how to present that to the user.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = fetcher.fetch_from_folder(images_dir)

    report = validator.validate(records)
    if not report.passed:
        # Validation failures are warnings, not hard stops, except for
        # outright rejected (too-low-resolution) images -- the caller can
        # inspect `report` for details before deciding whether to proceed.
        for w in report.warnings:
            typer.echo(f"[validator] warning: {w}")

    colmap_project = output_dir / "colmap"
    colmap_result = colmap_runner.run_sfm(
        image_dir=images_dir,
        project_dir=colmap_project,
        use_gpu=use_gpu,
    )

    splat_result = engine.train_splat(
        colmap_project_dir=colmap_result.sparse_dir,
        output_dir=output_dir / "splat",
        iterations=iterations,
        binary_path=opensplat_binary,
    )

    html_path = viewer.generate_viewer_html(
        splat_path=splat_result.ply_path,
        output_html=output_dir / "viewer.html",
        title="SceneForge Scene",
    )

    return PipelineResult(
        html_path=html_path,
        ply_path=splat_result.ply_path,
        num_images=report.image_count,
    )


@app.command()
def run(
    images: str = typer.Option(..., "--images", help="Folder of input images"),
    out: str = typer.Option("./sceneforge_output", "--out", help="Output directory"),
    iterations: int = typer.Option(1000, "--iterations", help="OpenSplat training iterations"),
    gpu: bool = typer.Option(False, "--gpu", help="Use GPU acceleration if available"),
):
    """Run the full pipeline: images -> COLMAP -> OpenSplat -> HTML viewer."""
    try:
        result = run_pipeline(images, out, iterations=iterations, use_gpu=gpu)
    except Exception as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Done. {result.num_images} images processed.")
    typer.echo(f"Splat: {result.ply_path}")
    typer.echo(f"Viewer: {result.html_path}")


if __name__ == "__main__":
    app()
