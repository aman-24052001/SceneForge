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
