# Docker — Status & Usage

## ⚠️ Honest status: NOT build-verified

The `Dockerfile` in this repo was written carefully against:
- COLMAP's apt package (confirmed working — this project installed COLMAP
  via `apt-get install colmap` and ran real reconstructions with it, see
  `docs/test_log.md`).
- OpenSplat's documented CPU build steps (CMake + libtorch CPU wheel),
  cross-referenced against OpenSplat's own (CUDA-only) Dockerfile in their
  repo and multiple independent real-world libtorch download examples.

**It has not actually been built.** The development sandbox used for this
project had no Docker daemon, and the libtorch CPU zip download
(`download.pytorch.org`) was blocked by that sandbox's network allowlist
(confirmed via a direct `curl` test — see `docs/test_log.md`). This is an
environment limitation, not a reason to believe the Dockerfile is wrong,
but it does mean **you are the first real build** if you run this.

## If the build fails

Most likely failure points, in order of likelihood:

1. **libtorch URL 404s.** PyTorch occasionally reshuffles old version
   downloads. Check https://pytorch.org/get-started/locally/ → LibTorch →
   CPU for the current URL pattern, and update `TORCH_VERSION` (or the URL
   itself) in the Dockerfile accordingly.
2. **OpenCV ABI mismatch.** OpenSplat was built/tested against specific
   OpenCV versions; apt's `libopencv-dev` on a given Ubuntu release may
   drift from that. If CMake configuration fails on OpenCV, check
   OpenSplat's GitHub issues for the Ubuntu version you're using.
3. **Ninja generator issues.** If `-GNinja` causes problems, fall back to
   the plain Makefile generator (remove `-GNinja` and `ninja`, use
   `cmake .. && make -j$(nproc)` instead, matching the OpenSplat README's
   simplest documented path).

Please open an issue with the exact error if you hit one of these — this
section should get more specific as real build attempts happen.

## Usage (once built)

```bash
docker build -t sceneforge .

docker run \
  -v $(pwd)/my_images:/data/images \
  -v $(pwd)/output:/data/output \
  sceneforge run --images /data/images --out /data/output --iterations 1000
```

Mount your images folder and an output folder as volumes; the container's
`ENTRYPOINT` runs `cli.py` directly, so any `cli.py run` flag works after
the image name (e.g. `--gpu`, `--force`, `--skip-quality-gate`).

## Why no multi-stage build (yet)

The current Dockerfile builds OpenSplat from source inside the final image,
which means build tools (cmake, ninja, git) and intermediate build
artifacts remain in the final image layer, making it larger than
necessary. A multi-stage build (compile in one stage, copy only the
`opensplat` binary into a slim final stage) would reduce image size
significantly. This is a reasonable follow-up once the single-stage build
is confirmed working end-to-end — optimizing an unverified build first
would make debugging harder.
