# SceneForge -- CPU-only image (no GPU/CUDA required)
#
# Builds COLMAP (via apt) and OpenSplat (from source, CPU libtorch) into
# a single image alongside the SceneForge Python pipeline.
#
# ⚠️ NOT YET BUILD-VERIFIED. This Dockerfile was written against the
# documented build steps for COLMAP (apt) and OpenSplat (CMake + libtorch
# CPU wheel, see https://github.com/pierotofy/OpenSplat#cpu), adapted from
# OpenSplat's own (CUDA-only) Dockerfile in their repo. It could not be
# built/tested in the development sandbox used for this project: no Docker
# daemon was available, and the libtorch CPU download below requires
# download.pytorch.org, which was blocked by that sandbox's network
# allowlist. See docs/test_log.md for the full account.
#
# If you build this and hit issues, the most likely culprits are:
#   - libtorch CPU zip URL/version mismatch (check https://pytorch.org/get-started/locally/
#     for the current CPU libtorch download URL if this one 404s)
#   - OpenCV version drift between apt's libopencv-dev and what OpenSplat expects
#
# Build:   docker build -t sceneforge .
# Run:     docker run -v $(pwd)/my_images:/data/images -v $(pwd)/output:/data/output \
#              sceneforge run --images /data/images --out /data/output

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

ARG TORCH_VERSION=2.2.1
ARG CMAKE_BUILD_TYPE=Release

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

WORKDIR /build

# --- System dependencies: COLMAP + OpenSplat build deps + Python ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap \
    build-essential \
    cmake \
    git \
    ninja-build \
    libopencv-dev \
    unzip \
    wget \
    python3 \
    python3-pip \
    && apt-get autoremove -y --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- libtorch (CPU build, no CUDA) ---
# URL pattern per https://pytorch.org/get-started/locally/ -> LibTorch -> CPU,
# and confirmed against multiple real-world build scripts using this exact
# pattern (e.g. PLUMED, AMReX docs). Pin a version here rather than tracking
# "latest" so builds are reproducible.
RUN wget --no-check-certificate -nv \
    "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cpu.zip" \
    -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt && \
    rm libtorch.zip

# --- OpenSplat (CPU build from source) ---
RUN git clone --depth 1 https://github.com/pierotofy/OpenSplat.git /build/OpenSplat && \
    cd /build/OpenSplat && \
    mkdir build && cd build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_PREFIX_PATH=/opt/libtorch \
        .. && \
    ninja && \
    cp opensplat /usr/local/bin/opensplat && \
    cd / && rm -rf /build/OpenSplat

# --- SceneForge Python pipeline ---
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY sceneforge/ ./sceneforge/
COPY cli.py .
COPY scripts/ ./scripts/

# Default working directories for mounted volumes
RUN mkdir -p /data/images /data/output

ENTRYPOINT ["python3", "cli.py"]
CMD ["--help"]
