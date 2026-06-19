"""
Orchestrator Module
Responsibilities:
- Glue fetcher -> validator -> colmap_runner -> engine -> viewer end-to-end
- Gate the expensive mapper/training stages behind quality + runtime checks:
    1. fetcher + validator (pre-input checks)
    2. colmap_runner.run_feature_matching (checkpointed)
    3. validator.validate_match_quality (fails fast, before mapper)
    4. colmap_runner.run_mapping (checkpointed)
    5. engine.estimate_runtime + warn (before training)
    6. engine.train_splat (checkpointed)
    7. viewer.generate_viewer_html
- Expose a CLI via Typer: `sceneforge run --images <folder> --out <folder>`
- All checkpointed stages mean re-invoking `run` after a crash resumes
  from the last completed stage rather than starting over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import typer

from sceneforge import colmap_runner, engine, fetcher, validator, viewer

app = typer.Typer(help="SceneForge: local images -> navigable 3D Gaussian Splat scene.")


class PipelineAborted(Exception):
    """Raised when a quality gate fails and the pipeline stops before the
    expensive stages (mapper, training) rather than continuing on doomed input."""


@dataclass
class PipelineResult:
    html_path: Path
    ply_path: Path
    num_images: int
    match_quality: validator.MatchQualityReport
    runtime_estimate: engine.RuntimeEstimate
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    images_dir: str | Path,
    output_dir: str | Path,
    iterations: int = 1000,
    use_gpu: bool = False,
    opensplat_binary: str | None = None,
    force: bool = False,
    skip_quality_gate: bool = False,
) -> PipelineResult:
    """
    Run the full SceneForge pipeline end-to-end.

    Args:
        images_dir: folder of input images.
        output_dir: where all outputs (colmap project, splat, viewer) go.
        iterations: OpenSplat training iteration count.
        use_gpu: whether to request GPU acceleration from COLMAP/OpenSplat.
        opensplat_binary: optional explicit path to the opensplat executable.
        force: re-run every stage even if checkpointed outputs exist.
        skip_quality_gate: proceed to mapper even if match quality looks bad
                            (useful for debugging; not recommended otherwise).

    Raises:
        fetcher.FetchError, colmap_runner.ColmapStageError/NotFoundError,
        engine.OpenSplatNotFoundError/RunError, or PipelineAborted if the
        post-matching quality gate fails and skip_quality_gate=False.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # --- Stage 1-2: fetch + pre-input validation ---
    records = fetcher.fetch_from_folder(images_dir)
    report = validator.validate(records)
    for w in report.warnings:
        typer.echo(f"[validator] warning: {w}")
        warnings.append(w)

    # --- Stage 3: feature extraction + matching (checkpointed) ---
    colmap_project = output_dir / "colmap"
    matching_result = colmap_runner.run_feature_matching(
        image_dir=images_dir,
        project_dir=colmap_project,
        use_gpu=use_gpu,
        force=force,
    )

    # --- Stage 4: post-matching quality gate ---
    # This is the check that would have caught the flat-cube texture
    # failure mode discovered during development (see docs/test_log.md)
    # BEFORE burning CPU time on the mapper stage.
    match_quality = validator.validate_match_quality(matching_result.database_path)
    for w in match_quality.warnings:
        typer.echo(f"[match_quality] warning: {w}")
        warnings.append(w)

    if not match_quality.passed and not skip_quality_gate:
        raise PipelineAborted(
            "Match quality gate failed -- stopping before the expensive mapper "
            f"stage. Best pair had {match_quality.best_pair_inliers} verified "
            "inliers. See warnings above. Pass skip_quality_gate=True to "
            "override (not recommended)."
        )

    # --- Stage 5: mapping / sparse reconstruction (checkpointed) ---
    colmap_result = colmap_runner.run_mapping(
        image_dir=images_dir,
        matching_result=matching_result,
        force=force,
    )

    # --- Stage 6: runtime estimate, surfaced BEFORE training starts ---
    runtime_estimate = engine.estimate_runtime(iterations=iterations, num_images=report.image_count)
    if runtime_estimate.warning:
        typer.echo(f"[engine] {runtime_estimate.warning}")
        warnings.append(runtime_estimate.warning)

    # --- Stage 7: training (checkpointed) ---
    splat_result = engine.train_splat(
        colmap_project_dir=colmap_result.sparse_dir,
        output_dir=output_dir / "splat",
        iterations=iterations,
        binary_path=opensplat_binary,
        force=force,
    )

    # --- Stage 8: viewer ---
    html_path = viewer.generate_viewer_html(
        splat_path=splat_result.ply_path,
        output_html=output_dir / "viewer.html",
        title="SceneForge Scene",
    )

    return PipelineResult(
        html_path=html_path,
        ply_path=splat_result.ply_path,
        num_images=report.image_count,
        match_quality=match_quality,
        runtime_estimate=runtime_estimate,
        warnings=warnings,
    )


@app.command()
def run(
    images: str = typer.Option(..., "--images", help="Folder of input images"),
    out: str = typer.Option("./sceneforge_output", "--out", help="Output directory"),
    iterations: int = typer.Option(1000, "--iterations", help="OpenSplat training iterations"),
    gpu: bool = typer.Option(False, "--gpu", help="Use GPU acceleration if available"),
    force: bool = typer.Option(False, "--force", help="Re-run all stages, ignoring checkpoints"),
    skip_quality_gate: bool = typer.Option(
        False, "--skip-quality-gate",
        help="Proceed to mapper even if match quality looks degenerate (debugging only)",
    ),
):
    """Run the full pipeline: images -> COLMAP -> OpenSplat -> HTML viewer."""
    try:
        result = run_pipeline(
            images, out,
            iterations=iterations,
            use_gpu=gpu,
            force=force,
            skip_quality_gate=skip_quality_gate,
        )
    except Exception as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Done. {result.num_images} images processed.")
    typer.echo(
        f"Match quality: {result.match_quality.pairs_with_inliers}/"
        f"{result.match_quality.total_pairs} pairs matched, "
        f"best pair {result.match_quality.best_pair_inliers} inliers."
    )
    typer.echo(f"Splat: {result.ply_path}")
    typer.echo(f"Viewer: {result.html_path}")
    if result.warnings:
        typer.echo(f"({len(result.warnings)} warnings were logged above)")


if __name__ == "__main__":
    app()
