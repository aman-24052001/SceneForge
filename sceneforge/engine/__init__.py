"""
Engine Module (OpenSplat -- free, CPU-capable 3DGS trainer)
Responsibilities:
- Run the OpenSplat CLI against a COLMAP project directory
- Produce a splat.ply (+ cameras.json) output
- Works without a GPU (~100x slower than CUDA, per upstream docs);
  keep -n (iteration count) low for CPU runs.

Reference: https://github.com/pierotofy/OpenSplat
  ./opensplat <colmap_project_dir> -n 2000        # default
  ./opensplat <colmap_project_dir> -n 1000 -o out.splat   # compressed + fewer iters
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class OpenSplatNotFoundError(Exception):
    """Raised when the opensplat binary isn't on PATH or at the given location."""


class OpenSplatRunError(Exception):
    def __init__(self, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"OpenSplat exited with code {returncode}: {stderr[:500]}")


@dataclass
class SplatResult:
    ply_path: Path
    cameras_json_path: Path | None


def _resolve_binary(binary_path: str | None) -> str:
    candidate = binary_path or shutil.which("opensplat")
    if not candidate:
        raise OpenSplatNotFoundError(
            "opensplat binary not found. Build from source: "
            "https://github.com/pierotofy/OpenSplat#build "
            "(CPU build requires only OpenCV + libtorch CPU wheel, no CUDA)."
        )
    return candidate


def train_splat(
    colmap_project_dir: str | Path,
    output_dir: str | Path,
    iterations: int = 1000,
    binary_path: str | None = None,
) -> SplatResult:
    """
    Run OpenSplat against a COLMAP project to produce a Gaussian Splat PLY.

    Args:
        colmap_project_dir: directory containing COLMAP's sparse/ model
                             (output of colmap_runner.run_sfm).
        output_dir: where to write splat.ply and cameras.json.
        iterations: training iteration count. Lower (e.g. 1000) for CPU
                    runs to keep wall-clock time reasonable; OpenSplat's
                    own default is 2000+ for GPU runs.
        binary_path: optional explicit path to the opensplat executable.

    Returns:
        SplatResult with paths to the generated assets.
    """
    binary = _resolve_binary(binary_path)

    colmap_project_dir = Path(colmap_project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary,
        str(colmap_project_dir),
        "-n", str(iterations),
        "-o", str(output_dir / "splat.ply"),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
    if proc.returncode != 0:
        raise OpenSplatRunError(proc.returncode, proc.stderr)

    ply_path = output_dir / "splat.ply"
    cameras_json = output_dir / "cameras.json"

    return SplatResult(
        ply_path=ply_path,
        cameras_json_path=cameras_json if cameras_json.exists() else None,
    )
