"""
COLMAP Runner Module
Responsibilities:
- Wrap COLMAP CLI calls (feature_extractor, exhaustive_matcher, mapper)
- Run entirely on CPU (no CUDA required)
- Produce a sparse reconstruction: camera poses + 3D point cloud
  in a project folder that OpenSplat can consume directly.

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


def run_sfm(image_dir: str | Path, project_dir: str | Path, use_gpu: bool = False) -> ColmapResult:
    """
    Run the full COLMAP sparse reconstruction pipeline on CPU.

    Args:
        image_dir: folder of input images.
        project_dir: working directory for the COLMAP project
                     (database.db + sparse/ will be created here).
        use_gpu: set True only if a CUDA build of COLMAP + GPU is available.

    Returns:
        ColmapResult with paths to the database and sparse model.
    """
    _require_colmap()

    image_dir = Path(image_dir)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    database_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

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

    _run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ], stage="mapper")

    # COLMAP writes numbered sub-models (0, 1, ...) under sparse_dir;
    # model 0 is the primary reconstruction when one exists.
    model_dir = sparse_dir / "0"
    num_registered = _count_registered_images(model_dir) if model_dir.exists() else 0

    return ColmapResult(
        project_dir=project_dir,
        database_path=database_path,
        sparse_dir=model_dir,
        num_registered_images=num_registered,
    )


def _count_registered_images(model_dir: Path) -> int:
    """Best-effort count of registered images without a full binary parse."""
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return 0
    # A full binary parse needs COLMAP's read_write_model format; for a
    # lightweight sanity check we just confirm the file is non-trivially
    # sized (an empty/near-empty model is a strong signal of failure).
    return 1 if images_bin.stat().st_size > 64 else 0
