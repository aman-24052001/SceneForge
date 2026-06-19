"""
MCP Server Module
Responsibilities:
- Expose SceneForge's pipeline as MCP tools, so an agent can drive
  image -> 3D-scene generation as a step in a larger workflow, instead of
  only via shelling out to cli.py.

Design choice -- async, non-blocking by default:
  The pipeline can take minutes (COLMAP) to hours (CPU OpenSplat training)
  to run. A tool that blocks until completion would hang the calling
  agent's turn for that entire duration, which is a poor MCP experience
  (no progress visibility, no way to do anything else meanwhile).

  Instead: `start_pipeline` launches the run in a background thread and
  returns immediately with a job_id. `check_job_status` polls progress.
  `get_viewer_path` retrieves the final output location once done. This
  mirrors the same poll-based pattern used by cloud 3DGS APIs (e.g. Luma
  AI's job/slug + status model) -- a pattern this project's research phase
  identified as the right shape for long-running generation tasks.

Tools exposed:
  - start_pipeline(images_dir, output_dir, iterations, use_gpu) -> job_id
  - check_job_status(job_id) -> status, stage, warnings
  - get_viewer_path(job_id) -> path to viewer.html (once completed)
  - list_jobs() -> all known job_ids and their status
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from sceneforge import engine, orchestrator, validator

mcp = FastMCP("sceneforge")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"  # quality gate stopped it


@dataclass
class Job:
    job_id: str
    images_dir: str
    output_dir: str
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    viewer_path: Optional[str] = None
    ply_path: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


# In-memory job registry. Fine for a single-process MCP server; if this
# needs to survive server restarts, swap this for a small SQLite table.
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _run_job(job: Job, iterations: int, use_gpu: bool, force: bool, skip_quality_gate: bool) -> None:
    job.status = JobStatus.RUNNING
    try:
        result = orchestrator.run_pipeline(
            images_dir=job.images_dir,
            output_dir=job.output_dir,
            iterations=iterations,
            use_gpu=use_gpu,
            force=force,
            skip_quality_gate=skip_quality_gate,
        )
        job.viewer_path = str(result.html_path)
        job.ply_path = str(result.ply_path)
        job.warnings = result.warnings
        job.status = JobStatus.COMPLETED
    except orchestrator.PipelineAborted as exc:
        job.status = JobStatus.ABORTED
        job.error = str(exc)
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the agent, don't swallow
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished_at = time.time()


@mcp.tool()
def start_pipeline(
    images_dir: str,
    output_dir: str = "./sceneforge_output",
    iterations: int = 1000,
    use_gpu: bool = False,
    force: bool = False,
    skip_quality_gate: bool = False,
) -> dict:
    """
    Start the SceneForge pipeline (fetcher -> validator -> COLMAP ->
    OpenSplat -> viewer) in the background and return immediately.

    This does NOT block -- use check_job_status(job_id) to poll progress.
    The pipeline can take from minutes (COLMAP only) to hours (CPU
    OpenSplat training) depending on image count, iterations, and
    whether a GPU build of OpenSplat is available.

    Args:
        images_dir: folder of input images on disk.
        output_dir: where to write colmap/, splat/, and viewer.html.
        iterations: OpenSplat training iteration count.
        use_gpu: request GPU acceleration if COLMAP/OpenSplat support it.
        force: ignore checkpoints and re-run every stage from scratch.
        skip_quality_gate: proceed past a degenerate match-quality result
                            instead of aborting (debugging only).

    Returns:
        dict with job_id to pass to check_job_status / get_viewer_path.
    """
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, images_dir=images_dir, output_dir=output_dir)

    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        args=(job, iterations, use_gpu, force, skip_quality_gate),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": job.status.value}


@mcp.tool()
def check_job_status(job_id: str) -> dict:
    """
    Poll the status of a pipeline job started with start_pipeline.

    Returns:
        dict with status (pending/running/completed/failed/aborted),
        error message if applicable, and any warnings logged during
        the run (e.g. low match quality, no GPU detected, too few images).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return {"error": f"No job found with id {job_id}"}

    elapsed = (job.finished_at or time.time()) - job.started_at
    return {
        "job_id": job_id,
        "status": job.status.value,
        "elapsed_seconds": round(elapsed, 1),
        "error": job.error,
        "warnings": job.warnings,
        "viewer_path": job.viewer_path,
        "ply_path": job.ply_path,
    }


@mcp.tool()
def get_viewer_path(job_id: str) -> dict:
    """
    Get the path to the generated viewer.html for a completed job.

    Returns an error if the job hasn't completed yet -- call
    check_job_status first to confirm status == "completed".
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return {"error": f"No job found with id {job_id}"}
    if job.status != JobStatus.COMPLETED:
        return {"error": f"Job is not completed yet (status: {job.status.value})"}

    return {"viewer_path": job.viewer_path, "ply_path": job.ply_path}


@mcp.tool()
def list_jobs() -> dict:
    """List all known jobs (this server process's lifetime only) and their status."""
    with _jobs_lock:
        return {
            jid: {"status": j.status.value, "images_dir": j.images_dir, "output_dir": j.output_dir}
            for jid, j in _jobs.items()
        }


@mcp.tool()
def check_environment() -> dict:
    """
    Check whether COLMAP and OpenSplat binaries are available, and whether
    a GPU is detected. Useful for an agent to sanity-check the environment
    before calling start_pipeline, rather than discovering a missing
    binary partway through a long run.
    """
    import shutil

    colmap_available = shutil.which("colmap") is not None
    opensplat_available = shutil.which("opensplat") is not None
    gpu_available = engine.detect_gpu()

    return {
        "colmap_available": colmap_available,
        "opensplat_available": opensplat_available,
        "gpu_available": gpu_available,
        "ready_to_run": colmap_available and opensplat_available,
    }


def main() -> None:
    """Entry point for running this as a standalone MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
