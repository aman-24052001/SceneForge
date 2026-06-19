---
name: Build report (Dockerfile / OpenSplat)
about: You attempted to build the Dockerfile or OpenSplat from source -- success or failure, please share!
title: "[BUILD] "
labels: build
---

**The Dockerfile in this repo is unverified (see docs/docker.md) -- your
build attempt, whether it succeeds or fails, is genuinely useful data.**

**Did the build succeed?**
- [ ] Yes, fully
- [ ] Partially (got past some steps, failed later)
- [ ] No

**Which step failed (if any)?**
- [ ] apt install (colmap, build deps)
- [ ] libtorch download
- [ ] OpenSplat CMake configure
- [ ] OpenSplat build (ninja/make)
- [ ] pip install (Python deps)
- [ ] Other

**Full error output**
```
paste here
```

**Environment**
- Docker version:
- Host OS:
- Architecture (x86_64 / arm64):
