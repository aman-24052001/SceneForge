"""
Viewer Module
Responsibilities:
- Generate a self-contained HTML page that loads a PLY/SPLAT file using
  @sparkjsdev/spark (Three.js-based renderer, loaded from CDN -- no
  npm build step required).
- Output a single .html file the user can open directly in a browser.
"""
from __future__ import annotations

from pathlib import Path

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ margin: 0; background: #0a0a0f; overflow: hidden; }}
  #info {{
    position: absolute; top: 12px; left: 12px; color: #ddd;
    font-family: monospace; font-size: 13px; z-index: 10;
    background: rgba(0,0,0,0.5); padding: 6px 10px; border-radius: 6px;
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
    "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/2.1.0/spark.module.js"
  }}
}}
</script>
</head>
<body>
<div id="info">{title} -- drag to orbit, scroll to zoom</div>
<script type="module">
  import * as THREE from "three";
  import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
  import {{ SparkRenderer, SplatMesh }} from "@sparkjsdev/spark";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    60, window.innerWidth / window.innerHeight, 0.01, 1000
  );
  camera.position.set(0, 0, 5);

  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const spark = new SparkRenderer({{ renderer }});
  scene.add(spark);

  const splatMesh = new SplatMesh({{ url: "{splat_url}" }});
  scene.add(splatMesh);

  window.addEventListener("resize", () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();
</script>
</body>
</html>
"""


def generate_viewer_html(splat_path: str | Path, output_html: str | Path, title: str = "SceneForge Viewer") -> Path:
    """
    Generate a standalone HTML viewer for a given PLY/SPLAT file.

    Args:
        splat_path: path to the .ply or .splat file. If it lives alongside
                    the output HTML, a relative path is used so the page
                    is portable.
        output_html: where to write the generated .html file.
        title: page title / on-screen label.

    Returns:
        Path to the written HTML file.
    """
    splat_path = Path(splat_path)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    try:
        splat_url = splat_path.relative_to(output_html.parent)
    except ValueError:
        splat_url = splat_path  # fall back to absolute/whatever was given

    html = _TEMPLATE.format(title=title, splat_url=splat_url.as_posix())
    output_html.write_text(html, encoding="utf-8")
    return output_html
