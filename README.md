# SceneForge 🌍

**Landmark-to-3D navigable environment pipeline using Gaussian Splatting.**

Input a landmark name → get a browser-navigable 3D scene. No local GPU required.

---

## Pipeline

```
Landmark Name / URL
      ↓
Image Fetcher          (Google Street View / Flickr API)
      ↓
Image Validator        (quality, coverage, overlap checks)
      ↓
Luma AI API            (3DGS reconstruction, cloud GPU)
      ↓
Polling Service        (async job tracking)
      ↓
Asset Downloader       (.splat / .glb)
      ↓
Browser Viewer         (Three.js Gaussian Splat renderer)
```

---

## Project Structure

```
SceneForge/
├── sceneforge/
│   ├── fetcher/        # Image sourcing (Street View, Flickr, WikiMedia)
│   ├── validator/      # Image quality & coverage checks
│   ├── engine/         # Luma AI API integration + polling
│   ├── viewer/         # Three.js browser viewer
│   └── orchestrator/   # End-to-end pipeline glue
├── docs/               # Architecture, API references, decisions
├── assets/
│   └── test_images/    # Sample inputs for dev/testing
├── requirements.txt
└── .env.example
```

---

## Stack

| Layer | Tool |
|---|---|
| Image Fetching | Google Street View API / Flickr API |
| 3DGS Reconstruction | Luma AI API |
| Rendering | Three.js + gaussian-splats-3d |
| Orchestration | Python (async) |

---

## Setup

```bash
git clone https://github.com/aman-24052001/SceneForge.git
cd SceneForge
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys
```

---

## Status

🚧 Phase 1 — Image Fetcher (in progress)

## License

MIT
