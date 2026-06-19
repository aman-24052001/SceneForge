"""
COLMAP Runner Module
Responsibilities:
- Wrap COLMAP CLI calls (feature_extractor, exhaustive_matcher, mapper)
- Run entirely on CPU (no CUDA required)
- Produce a sparse reconstruction: camera poses + 3D point cloud
  in a project folder that OpenSplat can consume directly.
- Split into two checkpointed stages (matching, then mapping) so the
  orchestrator can run validator.validate_match_quality() in between --
  catching degenerate inputs BEFORE burning CPU time on the expensive
  mapper stage.
- Each stage skips re-running if its output already exists, so a crashed
  pipeline can be re-invoked without redoing completed work.

Confirmed working via manual CLI testing in dev sandbox:
  colmap feature_extractor --image_path <imgs> --database_path <db> \
      --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1 \
      --SiftExtraction.use_gpu 0
  colmap exhaustive_matcher --database_path <db> --SiftMatching.use_gpu 0
  colmap mapper --database_path <db> --image_path <imgs> --output_path <sparse>
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ColmapNotFoundError(Exception):
    """Raised when the colmap binary isn't on PATH."""


class ColmapStageError(Exception):
    """Raised when a COLMAP CLI stage exits non-zero."""

    def __init__(self, stage: str, returncode: int, stderr: str):
        self.stage = stage
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"COLMAP stage '{stage}' failed (exit {returncode}): {stderr[:500]}")


@dataclass
class MatchingResult:
    project_dir: Path
    database_path: Path


@dataclass
class ColmapResult:
    project_dir: Path
    database_path: Path
    sparse_dir: Path
    num_registered_images: int


def _require_colmap() -> str:
    binary = shutil.which("colmap")
    if binary is None:
        raise ColmapNotFoundError(
            "colmap binary not found on PATH. Install with: apt-get install colmap"
        )
    return binary


def _run(cmd: list[str], stage: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ColmapStageError(stage, proc.returncode, proc.stderr)


def run_feature_matching(
    image_dir: str | Path,
    project_dir: str | Path,
    use_gpu: bool = False,
    force: bool = False,
) -> MatchingResult:
    """
    Run feature_extractor + exhaustive_matcher only (stops before the
    expensive mapper stage). Produces database.db, which the caller should
    inspect with validator.validate_match_quality() before proceeding to
    run_mapping().

    CHECKPOINT BEHAVIOR: if database.db already exists and is non-empty,
    this is skipped unless force=True. This lets a crashed/interrupted
    pipeline resume without redoing feature extraction + matching, which
    on CPU is the second-most expensive stage after mapping itself.

    Args:
        image_dir: folder of input images.
        project_dir: working directory for the COLMAP project.
        use_gpu: set True only if a CUDA build of COLMAP + GPU is available.
        force: re-run even if database.db already exists.

    Returns:
        MatchingResult with the path to the populated database.
    """
    _require_colmap()

    image_dir = Path(image_dir)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    database_path = project_dir / "database.db"

    if database_path.exists() and database_path.stat().st_size > 0 and not force:
        return MatchingResult(project_dir=project_dir, database_path=database_path)

    if database_path.exists() and force:
        database_path.unlink()

    gpu_flag = "1" if use_gpu else "0"

    _run([
        "colmap", "feature_extractor",
        "--image_path", str(image_dir),
        "--database_path", str(database_path),
        "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag,
    ], stage="feature_extractor")

    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ], stage="exhaustive_matcher")

    return MatchingResult(project_dir=project_dir, database_path=database_path)


def run_mapping(
    image_dir: str | Path,
    matching_result: MatchingResult,
    force: bool = False,
) -> ColmapResult:
    """
    Run the mapper stage against an already-matched database.

    CHECKPOINT BEHAVIOR: if sparse/0/images.bin already exists, this is
    skipped unless force=True.

    Args:
        image_dir: folder of input images (same one used for matching).
        matching_result: output of run_feature_matching().
        force: re-run even if a sparse model already exists.

    Returns:
        ColmapResult with paths to the database and sparse model.
    """
    _require_colmap()

    image_dir = Path(image_dir)
    project_dir = matching_result.project_dir
    database_path = matching_result.database_path
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    model_dir = sparse_dir / "0"

    already_done = model_dir.exists() and (model_dir / "images.bin").exists()
    if already_done and not force:
        num_registered = _count_registered_images(model_dir)
        return ColmapResult(
            project_dir=project_dir,
            database_path=database_path,
            sparse_dir=model_dir,
            num_registered_images=num_registered,
        )

    if already_done and force:
        shutil.rmtree(model_dir)

    _run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ], stage="mapper")

    num_registered = _count_registered_images(model_dir) if model_dir.exists() else 0

    return ColmapResult(
        project_dir=project_dir,
        database_path=database_path,
        sparse_dir=model_dir,
        num_registered_images=num_registered,
    )


def run_sfm(image_dir: str | Path, project_dir: str | Path, use_gpu: bool = False) -> ColmapResult:
    """
    Convenience wrapper: run matching + mapping back-to-back with no
    quality gate in between. Prefer calling run_feature_matching() and
    run_mapping() separately (via the orchestrator) so
    validator.validate_match_quality() can run between them.
    """
    matching_result = run_feature_matching(image_dir, project_dir, use_gpu=use_gpu)
    return run_mapping(image_dir, matching_result)


def _count_registered_images(model_dir: Path) -> int:
    """Best-effort count of registered images without a full binary parse."""
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return 0
    # A full binary parse needs COLMAP's read_write_model format; for a
    # lightweight sanity check we just confirm the file is non-trivially
    # sized (an empty/near-empty model is a strong signal of failure).
    return 1 if images_bin.stat().st_size > 64 else 0
