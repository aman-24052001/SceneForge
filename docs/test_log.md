# Test Log — What Was Actually Verified

This documents exactly what was tested, how, and what wasn't, so future
work (or future Claude sessions) don't have to re-derive this.

## Environment constraints discovered

- Dev sandbox: 1 CPU core, ~4GB RAM, ~5-10GB free disk.
- Network egress allowlist blocks `download.pytorch.org` (official libtorch
  source) — only `pypi.org`/`files.pythonhosted.org` are reachable.
- PyPI's default `torch` wheel bundles full CUDA (~3GB+), which alone can
  exhaust the available disk. A genuinely CPU-only torch wheel requires
  `download.pytorch.org`, which is blocked here.
- **Conclusion:** OpenSplat (which needs libtorch C++) could not be built
  in this sandbox. This is a sandbox limitation, not a project flaw — a
  normal dev machine with more disk and unrestricted network will not hit
  this.

## What WAS verified end-to-end

1. **COLMAP installs cleanly via apt** (`apt-get install colmap`), CPU-only
   build, confirmed via `colmap --help`.

2. **Synthetic test data generation** (`scripts/generate_test_scene.py`):
   - First attempt used a flat-faced cube — failed because flat checkerboard
     textures cause SIFT "repeated pattern" mismatches; RANSAC correctly
     rejected nearly all matches (0-16 inliers per pair).
   - Fixed by switching to a subdivided icosphere (genuine 3D curvature,
     320 triangles) with per-triangle unique noise texture (no repeating
     pattern). This produced 200+ verified geometric inliers per image pair.
   - Final generator: 24 views, orbiting camera, icosphere with backface
     culling and per-triangle fixed-seed texture (same patch looks
     consistent across views, as real-world texture does).

3. **Full COLMAP SfM pipeline ran successfully** on the 24 synthetic images:
   - `feature_extractor`: ~2700 SIFT features/image
   - `exhaustive_matcher`: 200+ verified inliers on strong pairs
   - `mapper`: produced a real sparse model (`images.bin` 261KB,
     `points3D.bin` 54KB) — non-trivial reconstruction, not an empty stub.

4. **All 6 Python modules** (`fetcher`, `validator`, `colmap_runner`,
   `engine`, `viewer`, `orchestrator`) pass `py_compile` syntax checks and
   import cleanly with correct cross-module references.

5. **Functional tests passed** (see `tests/test_modules.py`, 11/11 passing):
   - `fetcher.fetch_from_folder` correctly loads real images, raises
     `FetchError` on missing/empty folders.
   - `validator.validate` correctly passes 24 real images, warns on <10.
   - `viewer.generate_viewer_html` produces valid HTML referencing
     `SplatMesh` / Spark.js correctly.
   - `colmap_runner._require_colmap` finds the real installed binary, and
     raises `ColmapNotFoundError` cleanly when mocked as absent.
   - `engine._resolve_binary` raises `OpenSplatNotFoundError` cleanly
     (opensplat is genuinely not installed in this sandbox).
   - **`orchestrator.run_pipeline` chains all stages in the correct order**:
     it runs fetcher → validator → a REAL COLMAP reconstruction (proven by
     `images.bin`/`points3D.bin` existing on disk afterward) → and only
     then fails at the engine stage with the expected
     `OpenSplatNotFoundError`. This proves the wiring is correct, not just
     that each module works in isolation.

## What was NOT verified

- **OpenSplat training itself** (the actual Gaussian Splat optimization
  loop) was not run, since the binary couldn't be built in this sandbox.
  The CLI wrapper (`engine.train_splat`) is implemented per the documented
  OpenSplat CLI interface but its happy path is untested.
- **Viewer rendering in an actual browser** — the generated HTML's
  structure was checked (correct script tags, importmap, SplatMesh
  reference) but not opened in a real browser against a real PLY file.

## Recommended next step (on a machine with more resources)

```bash
git clone https://github.com/pierotofy/OpenSplat
cd OpenSplat && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
# Then:
python3 cli.py run --images assets/test_images --out ./output --iterations 1000
open ./output/viewer.html
```

---

## Update: gap-closing work (post initial-review)

Following an external review of this repo, the following gaps were closed
and re-tested. This section documents what changed and what's now verified.

### Validator: post-matching quality gate

Added `validator.validate_match_quality()`, which inspects COLMAP's
`database.db` directly (the `two_view_geometries` table) for verified
geometric inlier counts, **after** matching but **before** the expensive
mapper stage. This is exactly the check that would have caught the
flat-cube texture failure mode (0-16 inliers/pair) automatically instead
of requiring a manual SQLite query to diagnose.

Verified: `test_validate_match_quality_on_real_database` runs real COLMAP
matching against the icosphere test images and confirms the gate correctly
passes (200+ inliers on the best pair, well above the 50-inlier threshold).
Two more tests confirm it fails correctly on a missing or empty database.

### Checkpoint / resume support

`colmap_runner.run_feature_matching` and `run_mapping` were split apart
(previously one `run_sfm` call did both), each independently checkpointed:
if their output already exists on disk, the expensive subprocess call is
skipped unless `force=True`. Same pattern applied to `engine.train_splat`.

Verified: `test_colmap_runner_matching_is_checkpointed` confirms a second
call doesn't touch `database.db`'s mtime. `test_orchestrator_resumes_after_simulated_crash`
runs the full orchestrator twice against real COLMAP execution and confirms
the second run's COLMAP stages are skipped entirely (mtimes unchanged),
while the engine stage (which legitimately failed both times, since
opensplat isn't installed here) is correctly re-attempted.

### Runtime estimation + GPU detection

Added `engine.detect_gpu()` (checks for `nvidia-smi` on PATH) and
`engine.estimate_runtime()`, which surfaces a CPU-slowdown warning with a
rough time estimate **before** training starts, rather than the user
discovering 100x slower CPU performance partway into a run.

Verified: tests confirm the warning fires correctly when no GPU is
detected (true in this sandbox) and stays silent when one is.

### Orchestrator: quality gate wiring

`run_pipeline` now runs matching, checks quality, and only proceeds to
mapping if the gate passes (raising `PipelineAborted` otherwise, unless
`skip_quality_gate=True`).

Verified: `test_orchestrator_quality_gate_stops_before_mapper` mocks a bad
quality result and asserts `colmap_runner.run_mapping` is never called
(would raise `AssertionError` from the mock if it were) -- proving the gate
genuinely short-circuits, not just that it logs a warning.

### MCP server (`sceneforge/mcp_server/`)

Added a FastMCP-based server exposing 5 tools: `start_pipeline` (async,
returns immediately with a job_id), `check_job_status` (poll), 
`get_viewer_path`, `list_jobs`, and `check_environment`.

This is a genuinely new capability, not just a wrapper -- pipeline stages
run in a background thread, so an agent can start a job and continue doing
other things while it runs, rather than blocking for the pipeline's full
duration (minutes to hours on CPU).

Verified end-to-end against REAL COLMAP execution (not mocked):
- `test_mcp_start_pipeline_runs_async_and_reports_failure` confirms
  `start_pipeline()` returns in under 2 seconds (proving it's non-blocking)
  while COLMAP's real matching+mapping run in the background (confirmed by
  `images.bin` existing on disk afterward), then correctly reports
  `status: failed` with the exact `OpenSplatNotFoundError` message once the
  background thread reaches the (genuinely absent) engine stage.
- `test_mcp_check_environment_reflects_real_state` confirms
  `check_environment()` reports `colmap_available: true`,
  `opensplat_available: false` -- the actual state of this sandbox, not a
  hardcoded assumption.
- Error paths (`get_viewer_path` on unknown/incomplete jobs) confirmed.
- `PipelineAborted` (quality gate) confirmed to map to job status
  `"aborted"`, distinct from `"failed"` (genuine errors) -- tested by
  monkeypatching the quality check to simulate a degenerate scene and
  confirming `check_job_status` reports `aborted`.

### Still NOT verified (unchanged from before)

- OpenSplat training itself, end to end (still blocked by sandbox
  environment -- see original section above).
- The Dockerfile has not been built (no Docker daemon in this sandbox).
  See `docs/docker.md` for its honest status and likely failure points.
- The MCP server has not been tested against a real MCP client (e.g.
  Claude Desktop, Claude Code) -- only its tool registration, schemas, and
  underlying Python functions were tested directly. The stdio transport
  itself (`mcp.run()`) was not exercised.
