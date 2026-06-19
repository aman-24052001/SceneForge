"""
Generate a synthetic multi-view image set for testing the COLMAP SfM stage.

Renders a textured 3D cube (with distinct colored/patterned faces, so SIFT
features are distinguishable) from N camera positions on a circle around it.
This gives genuine multi-view parallax -- the same 3D points project to
different 2D pixel locations in each image, which is exactly what COLMAP's
feature matching + triangulation needs to solve for camera poses.

No external assets or downloads required -- pure numpy rasterization.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import math

OUT_DIR = "/home/claude/SceneForge/assets/test_images"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 480
FOCAL = 420  # pixels
N_VIEWS = 24
RADIUS = 4.0  # camera distance from object center
CUBE_HALF = 1.0

# 8 cube vertices
verts = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64) * CUBE_HALF

# 6 faces (vertex indices), each face gets a distinct color + a checker pattern
faces = [
    ([0, 1, 2, 3], (220, 60, 60)),    # back
    ([4, 5, 6, 7], (60, 200, 80)),    # front
    ([0, 1, 5, 4], (60, 90, 220)),    # bottom
    ([3, 2, 6, 7], (230, 200, 40)),   # top
    ([0, 3, 7, 4], (200, 90, 220)),   # left
    ([1, 2, 6, 5], (40, 210, 210)),   # right
]


def make_icosphere(subdivisions=2):
    """Build a unit icosphere mesh: returns (vertices, triangular faces)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    verts = [np.array(v) / np.linalg.norm(v) for v in verts]
    tris = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    def midpoint(cache, i1, i2):
        key = tuple(sorted((i1, i2)))
        if key in cache:
            return cache[key]
        mid = (verts[i1] + verts[i2]) / 2.0
        mid = mid / np.linalg.norm(mid)
        verts.append(mid)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(subdivisions):
        cache = {}
        new_tris = []
        for a, b, c in tris:
            ab = midpoint(cache, a, b)
            bc = midpoint(cache, b, c)
            ca = midpoint(cache, c, a)
            new_tris += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        tris = new_tris

    return [v * CUBE_HALF * 1.3 for v in verts], tris


ico_verts, ico_tris = make_icosphere(subdivisions=2)
# Assign each triangle a base color from a small distinct palette, grouped
# by spatial region so the surface reads as a coherent "textured terrain"
# rather than pure noise (helps both SIFT repeatability and visual sanity).
palette = [
    (210, 70, 70), (70, 200, 90), (70, 110, 220), (225, 200, 50),
    (200, 90, 220), (50, 200, 200), (230, 140, 60), (150, 150, 150),
]
tri_colors = [palette[i % len(palette)] for i in range(len(ico_tris))]


def look_at_matrix(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """Build a camera-to-world rotation matrix (right, up, forward)."""
    forward = (target - cam_pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # World-to-camera rotation: rows are camera axes
    R = np.stack([right, true_up, forward], axis=0)
    return R


def project(point_world, cam_pos, R, focal, img_size):
    """Project a 3D world point into 2D pixel coords using pinhole model."""
    p_cam = R @ (point_world - cam_pos)
    if p_cam[2] <= 0.01:
        return None  # behind camera
    x = focal * (p_cam[0] / p_cam[2]) + img_size / 2
    y = -focal * (p_cam[1] / p_cam[2]) + img_size / 2
    depth = p_cam[2]
    return (x, y, depth)


def textured_tri(draw, projected_2d, base_color, rng, grid=6):
    """
    Fill a triangle with blotchy non-repeating shading via barycentric
    subdivision. Same anti-aliasing-of-SIFT-matches rationale as before:
    unique-looking texture per surface patch, consistent across views.
    """
    p0, p1, p2 = projected_2d

    for gy in range(grid):
        for gx in range(grid - gy):
            # Two small triangles per barycentric cell (skip the diagonal split
            # for the last row to stay inside the triangle)
            def bary(u, v):
                w = 1 - u - v
                return (w * p0[0] + u * p1[0] + v * p2[0],
                        w * p0[1] + u * p1[1] + v * p2[1])

            u0, v0 = gx / grid, gy / grid
            u1, v1 = (gx + 1) / grid, gy / grid
            u2, v2 = gx / grid, (gy + 1) / grid
            cell = [bary(u0, v0), bary(u1, v1), bary(u2, v2)]
            offset = rng.integers(-60, 61)
            color = tuple(int(np.clip(c + offset, 0, 255)) for c in base_color)
            draw.polygon(cell, fill=color)


def render_view(cam_pos, R, focal, img_size, tri_seeds):
    bg_rng = np.random.default_rng(7)
    bg = bg_rng.integers(20, 60, size=(img_size, img_size, 3), dtype=np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img)

    tri_depths = []
    for ti, (a, b, c) in enumerate(ico_tris):
        v0, v1, v2 = ico_verts[a], ico_verts[b], ico_verts[c]
        # Backface cull: skip triangles facing away from the camera
        centroid = (v0 + v1 + v2) / 3.0
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        view_dir = cam_pos - centroid
        if np.dot(normal, view_dir) <= 0:
            continue
        p_cam = R @ (centroid - cam_pos)
        tri_depths.append((p_cam[2], a, b, c, ti))
    tri_depths.sort(key=lambda f: -f[0])  # farthest first

    for depth, a, b, c, ti in tri_depths:
        if depth <= 0:
            continue
        projected = []
        visible = True
        for vi in (a, b, c):
            res = project(ico_verts[vi], cam_pos, R, focal, img_size)
            if res is None:
                visible = False
                break
            projected.append((res[0], res[1]))
        if not visible:
            continue
        tri_rng = np.random.default_rng(tri_seeds[ti])
        textured_tri(draw, projected, tri_colors[ti], tri_rng, grid=6)

    return img


def main():
    print(f"Generating {N_VIEWS} synthetic multi-view images -> {OUT_DIR}")
    print(f"Icosphere: {len(ico_verts)} vertices, {len(ico_tris)} triangles")
    tri_seeds = [2000 + i for i in range(len(ico_tris))]  # fixed per-triangle texture identity

    for i in range(N_VIEWS):
        angle = 2 * math.pi * i / N_VIEWS
        # Orbit camera around the object at fixed radius, slight height variation
        cam_x = RADIUS * math.cos(angle)
        cam_z = RADIUS * math.sin(angle)
        cam_y = 1.2 * math.sin(angle * 2)  # bob up/down across the orbit
        cam_pos = np.array([cam_x, cam_y, cam_z])

        R = look_at_matrix(cam_pos)
        img = render_view(cam_pos, R, FOCAL, IMG_SIZE, tri_seeds)

        fname = os.path.join(OUT_DIR, f"view_{i:03d}.png")
        img.save(fname)
        print(f"  [{i+1}/{N_VIEWS}] {fname}  (cam_pos={cam_pos.round(2)})")

    print("Done.")


if __name__ == "__main__":
    main()
