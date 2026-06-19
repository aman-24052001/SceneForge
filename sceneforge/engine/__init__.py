"""
Engine Module (OpenSplat -- free, CPU-capable 3DGS trainer)
Responsibilities:
- Detect whether a GPU is actually available to the opensplat binary, and
  estimate a rough wall-clock time so the user isn't surprised by a CPU
  run that takes ~100x longer than they expected.
- Run the OpenSplat CLI against a COLMAP project directory.
- Produce a splat.ply (+ cameras.json) output.
- Skip re-training if splat.ply already exists (checkpoint behavior),
  unless force=True.

Reference: https://github.com/pierotofy/OpenSplat
  ./opensplat <colmap_project_dir> -n 2000        # default
  ./opensplat <colmap_project_dir> -n 1000 -o out.splat   # compressed + fewer iters

NOTE: training itself has not been run end-to-end in this project's dev
environment (libtorch build blocked by sandbox network/disk limits --
see docs/test_log.md). This wrapper is implemented against OpenSplat's
documented CLI interface but its happy path is unverified.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Rough CPU-vs-GPU slowdown factor as reported by the OpenSplat README.
# This is a documentation-derived estimate, not benchmarked by us --
# treat the resulting time estimate as an order-of-magnitude warning,
# not a precise prediction.
CPU_SLOWDOWN_FACTOR = 100
GPU_SECONDS_PER_ITERATION_ESTIMATE = 0.05  # rough, varies hugely by scene/GPU


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


@dataclass
class RuntimeEstimate:
    gpu_available: bool
    estimated_seconds: float
    warning: str | None


def _resolve_binary(binary_path: str | None) -> str:
    candidate = binary_path or shutil.which("opensplat")
    if not candidate:
        raise OpenSplatNotFoundError(
            "opensplat binary not found. Build from source: "
            "https://github.com/pierotofy/OpenSplat#build "
            "(CPU build requires only OpenCV + libtorch CPU wheel, no CUDA)."
        )
    return candidate


def detect_gpu() -> bool:
    """
    Best-effort GPU detection: checks for `nvidia-smi` (NVIDIA) on PATH.
    This does NOT guarantee the opensplat binary itself was built with
    GPU support -- a CPU-only build will still run on CPU even if a GPU
    is physically present. It's a heuristic for the runtime estimate only.
    """
    return shutil.which("nvidia-smi") is not None


def estimate_runtime(iterations: int, num_images: int) -> RuntimeEstimate:
    """
    Produce a rough wall-clock estimate and surface the CPU slowdown
    warning BEFORE the user kicks off a long run, rather than discovering
    it 40 minutes in.

    This is intentionally a coarse order-of-magnitude estimate, not a
    benchmark-backed prediction -- actual time depends heavily on scene
    complexity, point cloud density, and image resolution.
    """
    gpu = detect_gpu()
    base_seconds = iterations * GPU_SECONDS_PER_ITERATION_ESTIMATE
    # More images -> more Gaussians initialized -> slower per-iteration cost.
    # This scaling factor is a heuristic, not derived from real benchmarks.
    scale = max(1.0, num_images / 20)
    estimated = base_seconds * scale

    warning = None
    if not gpu:
        estimated *= CPU_SLOWDOWN_FACTOR
        warning = (
            f"No GPU detected (nvidia-smi not found). OpenSplat will run on CPU, "
            f"which is roughly {CPU_SLOWDOWN_FACTOR}x slower than GPU per the "
            f"upstream README. Estimated time: ~{estimated / 60:.0f} minutes for "
            f"{iterations} iterations. Consider reducing --iterations for a first test."
        )

    return RuntimeEstimate(gpu_available=gpu, estimated_seconds=estimated, warning=warning)


def train_splat(
    colmap_project_dir: str | Path,
    output_dir: str | Path,
    iterations: int = 1000,
    binary_path: str | None = None,
    force: bool = False,
) -> SplatResult:
    """
    Run OpenSplat against a COLMAP project to produce a Gaussian Splat PLY.

    CHECKPOINT BEHAVIOR: if output_dir/splat.ply already exists, training
    is skipped unless force=True. This is the single most expensive stage
    on CPU, so avoiding accidental re-runs matters most here.

    Args:
        colmap_project_dir: directory containing COLMAP's sparse/ model
                             (output of colmap_runner.run_mapping / run_sfm).
        output_dir: where to write splat.ply and cameras.json.
        iterations: training iteration count. Lower (e.g. 1000) for CPU
                    runs to keep wall-clock time reasonable; OpenSplat's
                    own default is 2000+ for GPU runs.
        binary_path: optional explicit path to the opensplat executable.
        force: re-run even if splat.ply already exists.

    Returns:
        SplatResult with paths to the generated assets.
    """
    binary = _resolve_binary(binary_path)

    colmap_project_dir = Path(colmap_project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = output_dir / "splat.ply"
    cameras_json = output_dir / "cameras.json"

    if ply_path.exists() and ply_path.stat().st_size > 0 and not force:
        return SplatResult(
            ply_path=ply_path,
            cameras_json_path=cameras_json if cameras_json.exists() else None,
        )

    cmd = [
        binary,
        str(colmap_project_dir),
        "-n", str(iterations),
        "-o", str(ply_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
    if proc.returncode != 0:
        raise OpenSplatRunError(proc.returncode, proc.stderr)

    return SplatResult(
        ply_path=ply_path,
        cameras_json_path=cameras_json if cameras_json.exists() else None,
    )
