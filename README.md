# SceneForge 🌍

**Local images → navigable 3D Gaussian Splat scene. 100% free, no API keys, CPU-capable.**

---

## Pipeline

```
Folder of images
      ↓
fetcher          -- load + validate image files (local, no API)
      ↓
validator        -- count / resolution / blur checks (pre-input)
      ↓
colmap_runner    -- COLMAP feature extraction + matching (checkpointed)
      ↓
validator        -- match-quality gate (post-matching, catches degenerate
                     scenes BEFORE the expensive mapper stage)
      ↓
colmap_runner    -- COLMAP mapper: camera poses + sparse 3D point cloud (checkpointed)
      ↓
engine           -- GPU detection + runtime estimate, then OpenSplat training (checkpointed)
      ↓
viewer           -- generates a self-contained HTML page (Spark.js/Three.js)
```

Also exposed as MCP tools (`sceneforge/mcp_server/`) for agentic/programmatic
use -- see `docs/mcp_server.md`.

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
│   ├── validator/       # pre-input checks + post-matching quality gate
│   ├── colmap_runner/   # wraps COLMAP CLI, checkpointed (matching + mapping split)
│   ├── engine/          # wraps OpenSplat CLI, GPU detection, runtime estimate, checkpointed
│   ├── viewer/          # generates the Spark.js HTML viewer
│   ├── orchestrator/    # chains all stages + Typer CLI + quality gate wiring
│   └── mcp_server/      # FastMCP server: async job-based pipeline tools
├── scripts/
│   └── generate_test_scene.py   # synthetic multi-view test image generator
├── tests/
│   └── test_modules.py          # 26 tests: wiring, unit, MCP, checkpoint/resume
├── assets/test_images/          # generated sample images (24 views of a textured icosphere)
├── .github/
│   ├── workflows/ci.yml         # CI: fast tests + real-COLMAP integration tests
│   └── ISSUE_TEMPLATE/
├── docs/
│   ├── architecture.md
│   ├── test_log.md              # what was actually verified, and how
│   ├── docker.md                 # Docker status (unverified -- see below)
│   └── mcp_server.md
├── Dockerfile                    # ⚠️ unverified, see docs/docker.md
├── pyproject.toml                # proper packaging (pip install -e .)
├── cli.py                        # convenience entrypoint
└── requirements.txt
```

---

## Setup

### Option A: pip
```bash
pip install -e .
```

### Option B: requirements.txt
```bash
pip install -r requirements.txt
```

### External binaries (required, not pip-installable)
```bash
# COLMAP (CPU build is fine, confirmed working)
apt-get install colmap

# OpenSplat (build from source -- NOT yet verified by this project, see docs/test_log.md)
git clone https://github.com/pierotofy/OpenSplat
cd OpenSplat && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```
See https://github.com/pierotofy/OpenSplat#build for full build instructions.

### Option C: Docker (⚠️ unverified build, see `docs/docker.md`)
```bash
docker build -t sceneforge .
docker run -v $(pwd)/images:/data/images -v $(pwd)/output:/data/output \
    sceneforge run --images /data/images --out /data/output
```

---

## Usage

### CLI
```bash
# Generate a synthetic test scene (no real photos needed)
python3 scripts/generate_test_scene.py

# Run the full pipeline (resumes from checkpoints if re-run after a crash)
python3 cli.py run --images assets/test_images --out ./output --iterations 1000

# Force re-run every stage, ignoring checkpoints
python3 cli.py run --images assets/test_images --out ./output --force

# Open the result
open ./output/viewer.html
```

### As an installed package
```bash
pip install -e .
sceneforge run --images assets/test_images --out ./output
```

### MCP server (agentic use)
```bash
sceneforge-mcp
```
See `docs/mcp_server.md` for tool reference and client config.

---

## Testing

```bash
python -m pytest tests/ -v
```

26 tests covering: module imports, fetcher/validator/viewer behavior,
error-handling paths, checkpoint/resume correctness, the post-matching
quality gate, and the MCP server's async job lifecycle — multiple of
which run **real COLMAP reconstructions** against the synthetic test
images, not just mocks. See `docs/test_log.md` for full detail on what
was verified and how.

CI runs on every push/PR (`.github/workflows/ci.yml`): fast unit tests,
a slower job that installs real COLMAP and runs the integration tests,
and a check that the test-scene generator still works.

---

## Status

✅ Fetcher, validator (pre + post-matching), colmap_runner (checkpointed),
   engine (checkpointed, GPU detection, runtime estimate), viewer,
   orchestrator (quality-gated, resumable), MCP server — implemented and tested

✅ Real COLMAP SfM verified end-to-end on synthetic test images (CPU),
   including the post-matching quality gate catching degenerate input

✅ MCP server's async job lifecycle verified against real COLMAP execution

⚠️ OpenSplat training itself not yet run end-to-end (needs a build
   environment with more disk/network access than this project's dev
   sandbox had — see `docs/test_log.md`)

⚠️ Dockerfile written but not build-verified (no Docker daemon available
   in dev sandbox — see `docs/docker.md`)

## Contributing

See `CONTRIBUTING.md`. The highest-priority open item is a real
end-to-end OpenSplat run on non-synthetic photos.

## License

MIT (project code). COLMAP is BSD-licensed; OpenSplat is AGPLv3 — see their
respective repos for terms if you redistribute binaries built from them.
This project's code does not currently enforce or check AGPL obligations
programmatically; that responsibility falls on anyone redistributing built
OpenSplat binaries themselves.
