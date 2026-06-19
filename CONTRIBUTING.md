# Contributing to SceneForge

## Before you start

This project is in active early development. The single biggest unverified
piece is **OpenSplat training itself never running end-to-end** — see
`docs/test_log.md` for the full honest status. If you're looking for a
high-impact first contribution, running the pipeline through to a real
splat.ply on real photos and reporting back (success or failure details)
is the most valuable thing you can do right now.

## Setup

```bash
git clone https://github.com/aman-24052001/SceneForge.git
cd SceneForge
pip install -r requirements.txt

# External binaries (see README.md for full instructions):
apt-get install colmap
# OpenSplat: build from source, see docs/docker.md and
# https://github.com/pierotofy/OpenSplat#build
```

## Running tests

```bash
python -m pytest tests/ -v
```

Note: several tests run real COLMAP reconstructions against the synthetic
test images in `assets/test_images/` and take 60-90+ seconds each on a
single CPU core. This is intentional — they're integration tests, not
pure unit tests. If you're iterating quickly, filter with `-k`:

```bash
# Fast tests only (no real COLMAP execution)
python -m pytest tests/ -v -k "fetcher or validator_passes or engine or viewer"
```

## Code style

- Type hints on all public function signatures.
- Dataclasses for structured return values (see `ColmapResult`,
  `SplatResult`, `ValidationReport` for the pattern).
- Every module-level exception type should be documented in its module's
  docstring and have at least one test confirming it's raised correctly
  (not just that the happy path works).
- Checkpointed stages (anything that wraps a slow subprocess call) should
  skip re-running if their output already exists, unless `force=True` is
  passed. See `colmap_runner.run_feature_matching` for the pattern.

## Reporting issues

Please include:
- What stage failed (fetcher / validator / colmap_runner / engine / viewer
  / orchestrator).
- Full error message/traceback.
- Whether you're running on CPU or GPU, and your OS.
- If it's an OpenSplat build issue, your libtorch/CMake/OpenCV versions.

## Priority areas (see repo's known-gaps discussion)

1. Real end-to-end OpenSplat run on non-synthetic photos.
2. Confirming/fixing the Dockerfile build (untested — see `docs/docker.md`).
3. Real-photo integration test (current tests use synthetic icosphere images).
4. MCP server wrapper for agentic/programmatic use beyond the CLI.

No formal PR template yet — just describe what changed and why, and link
which gap (if any) it addresses.
