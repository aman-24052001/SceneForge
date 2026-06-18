# SceneForge — Architecture

## Pipeline Stages

### Stage 1: Image Fetcher
- Input: landmark name (string) or direct image URL
- Sources: Google Street View Static API, Flickr API (free tier)
- Output: list of image file paths + metadata (lat/lng, heading, source)
- Target: 15–25 images with angular diversity

### Stage 2: Image Validator
- Min image count: 10
- Min resolution: 640×480
- Quality checks: blur detection, exposure
- Coverage check: heading spread heuristic
- Output: validated image list or rejection report

### Stage 3: Luma AI Engine
- Upload images via Luma AI Python SDK
- Create capture → trigger reconstruction
- Poll every N seconds for status
- On completion: download .splat or .glb

### Stage 4: Viewer
- Three.js page with gaussian-splats-3d library
- Load .splat asset
- WASD + mouse navigation
- Export as self-contained HTML

### Stage 5: Orchestrator
- CLI: `sceneforge run --landmark "Eiffel Tower"`
- Chains all stages with error handling
- Saves artifacts to output/ directory

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| No local GPU | Luma AI cloud | Feasibility |
| Image source | Street View + Flickr | Coverage + free fallback |
| Viewer | Three.js | Browser-native, no install |
| Async polling | httpx async | Non-blocking wait |
