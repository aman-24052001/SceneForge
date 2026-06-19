# SceneForge 🌍

**Local images → navigable 3D Gaussian Splat scene. 100% free, no API keys, CPU-capable.**

---

## Pipeline

```
Folder of images
      ↓
fetcher          -- load + validate image files (local, no API)
      ↓
validator        -- count / resolution / blur checks
      ↓
colmap_runner    -- COLMAP SfM: camera poses + sparse 3D point cloud (CPU)
      ↓
engine           -- OpenSplat: trains Gaussian Splat from COLMAP output (CPU or GPU)
      ↓
viewer           -- generates a self-contained HTML page (Spark.js/Three.js)
```

---

## Why this stack

| Concern | Choice | Why |
|---|---|---|
| Cost | $0 | No Luma AI / cloud API — fully self-hosted |
| GPU | Optional | OpenSplat runs on CPU (~100x slower, but works) |
| Camera poses | COLMAP | Free, open-source, apt-installable |
| Gaussian Splat training | OpenSplat | AGPLv3, CPU/GPU support, takes COLMAP output directly |
| Viewer | Spark.js (CDN) | No build step, drops into a single HTML file |

See `docs/architecture.md` for the full design rationale, including why
Luma AI and other paid cloud APIs were ruled out.

---

## Project Structure

```
SceneForge/
├── sceneforge/
│   ├── fetcher/         # load images from a local folder
│   ├── validator/       # quality/count checks
│   ├── colmap_runner/   # wraps the COLMAP CLI (feature_extractor, matcher, mapper)
│   ├── engine/          # wraps the OpenSplat CLI
│   ├── viewer/          # generates the Spark.js HTML viewer
│   └── orchestrator/    # chains all stages + Typer CLI
├── scripts/
│   └── generate_test_scene.py   # synthetic multi-view test image generator
├── tests/
│   └── test_modules.py          # wiring + unit tests (see docs/test_log.md)
├── assets/test_images/          # generated sample images (24 views of a textured icosphere)
├── docs/
│   ├── architecture.md
│   └── test_log.md              # what was actually verified, and how
├── cli.py                       # convenience entrypoint
└── requirements.txt
```

---

## Setup

### Python dependencies
```bash
pip install -r requirements.txt
```

### External binaries (required, not pip-installable)
```bash
# COLMAP (CPU build is fine)
apt-get install colmap

# OpenSplat (build from source)
git clone https://github.com/pierotofy/OpenSplat
cd OpenSplat && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```
See https://github.com/pierotofy/OpenSplat#build for full build instructions
(CPU build needs OpenCV dev + libtorch CPU wheel, no CUDA required).

---

## Usage

```bash
# Generate a synthetic test scene (no real photos needed)
python3 scripts/generate_test_scene.py

# Run the full pipeline
python3 cli.py run --images assets/test_images --out ./output --iterations 1000

# Open the result
open ./output/viewer.html
```

---

## Testing

```bash
python3 -m pytest tests/ -v
```

11 tests covering: module imports, fetcher/validator/viewer behavior on
real sample data, error-handling paths, and orchestrator stage-chaining
(verified against a real COLMAP run — see `docs/test_log.md`).

---

## Status

✅ Fetcher, validator, colmap_runner, engine, viewer, orchestrator — implemented and tested
✅ COLMAP SfM verified working end-to-end on synthetic test images (CPU)
⚠️ OpenSplat training not yet run end-to-end (needs build environment with libtorch; see docs/test_log.md)

## License

MIT (project code). COLMAP is BSD-licensed; OpenSplat is AGPLv3 — see their
respective repos for terms if you redistribute binaries built from them.
