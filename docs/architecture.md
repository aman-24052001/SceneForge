# SceneForge — Architecture

## Why this stack (history)

Earlier designs considered Luma AI (cloud 3DGS API) and Google Street View
(paid image source). Both were ruled out per explicit requirement: **zero
cost, no API keys**. Research findings that drove the final design:

- `graphdeco-inria/gaussian-splatting` (the original 3DGS paper's reference
  implementation) requires CUDA — no CPU fallback.
- `nerfstudio-project/gsplat` requires PyTorch + CUDA toolkit; falls back to
  CPU only in a degraded/experimental capacity.
- **`pierotofy/OpenSplat`** is the exception: explicitly supports CPU-only
  builds (~100x slower than GPU, but functional), AGPLv3 licensed,
  commercial use allowed. This became our engine.
- COLMAP itself has no CUDA requirement for the sparse reconstruction (SfM)
  stage used here — only its dense/MVS stage needs GPU, which we don't use.

## Pipeline Stages

### Stage 1: Fetcher (`sceneforge/fetcher`)
- Input: local folder path of images.
- No external API — intentionally offline-first.
- Output: list of `ImageRecord` (path, width, height).

### Stage 2: Validator (`sceneforge/validator`)
- Min image count: 10 (COLMAP needs >=3 technically, but reconstructions
  are unstable below ~10-15 in practice).
- Min resolution: 200px shorter side.
- Blur heuristic: Laplacian variance (implemented in pure numpy, no OpenCV
  dependency needed at this stage).

### Stage 3: COLMAP Runner (`sceneforge/colmap_runner`)
- Wraps three CLI calls: `feature_extractor` → `exhaustive_matcher` → `mapper`.
- Runs CPU-only (`--SiftExtraction.use_gpu 0`, `--SiftMatching.use_gpu 0`).
- Output: a COLMAP project directory with `database.db` and `sparse/0/`
  (camera poses + triangulated 3D points).

### Stage 4: Engine (`sceneforge/engine`)
- Wraps the OpenSplat CLI: `opensplat <colmap_project_dir> -n <iters> -o splat.ply`.
- CPU or GPU, depending on how the user built the OpenSplat binary.
- Output: `splat.ply` (+ `cameras.json`).

### Stage 5: Viewer (`sceneforge/viewer`)
- Generates a single self-contained HTML file.
- Uses `@sparkjsdev/spark` loaded from CDN (no npm/build step) — supports
  `.ply` directly, so no client-side conversion needed.

### Stage 6: Orchestrator (`sceneforge/orchestrator`)
- Chains stages 1-5 in order.
- Typer-based CLI: `python3 cli.py run --images <dir> --out <dir>`.
- Stage-specific exceptions propagate with clear messages rather than
  being swallowed — see `docs/test_log.md` for confirmation this chaining
  works correctly against a real COLMAP run.

## Math → Code Mapping

| Concept (see chat history for full derivation) | Where it lives |
|---|---|
| Camera pose estimation, bundle adjustment | COLMAP `mapper` (external binary) |
| 3D Gaussian (μ, Σ via R·S, opacity, SH color) | OpenSplat training loop (external binary) |
| PLY fields: `x,y,z`, `scale_*`, `rot_*`, `opacity`, `f_dc_*`, `f_rest_*` | `splat.ply` output, consumed as-is by the viewer |
| EWA projection + alpha compositing | Spark.js renderer (client-side, CDN) |

SceneForge's own Python code does **not** reimplement any of this math —
it orchestrates two well-tested external tools (COLMAP, OpenSplat) that
already implement it correctly, and adds the missing glue: input
validation, CLI wiring, and a zero-build viewer.

## Known Limitations

- CPU training is slow. For real (non-synthetic) scenes, expect this to be
  impractical without a GPU build of OpenSplat for anything beyond a small
  test object.
- COLMAP's incremental SfM needs genuine 3D structure in the scene to chain
  image registration past the seed pair — flat/low-curvature objects (e.g.
  a simple cube) are a known degenerate case. See `docs/test_log.md` for
  how this was discovered and fixed in the synthetic test generator.
