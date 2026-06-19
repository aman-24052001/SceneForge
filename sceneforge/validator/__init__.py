"""
Image Validator Module
Responsibilities:
- Check image count meets minimum threshold for SfM (COLMAP needs >= 3,
  realistically >= 10-15 for a stable reconstruction)
- Check minimum resolution
- Flag (not silently drop) low-quality images via blur heuristic
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from sceneforge.fetcher import ImageRecord

MIN_IMAGE_COUNT = 10
MIN_RESOLUTION = 200  # px, shorter side
BLUR_VARIANCE_THRESHOLD = 50.0  # Laplacian variance; lower = blurrier


@dataclass
class ValidationReport:
    passed: bool
    image_count: int
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Simple Laplacian-based blur metric without requiring OpenCV."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    # Valid-mode 2D convolution via numpy (small images, fine without scipy)
    out = np.zeros((h - 2, w - 2))
    for ky in range(3):
        for kx in range(3):
            out += kernel[ky, kx] * gray[ky:ky + h - 2, kx:kx + w - 2]
    return float(out.var())


def validate(records: list[ImageRecord]) -> ValidationReport:
    report = ValidationReport(passed=True, image_count=len(records))

    if len(records) < MIN_IMAGE_COUNT:
        report.passed = False
        report.warnings.append(
            f"Only {len(records)} images found; recommend >= {MIN_IMAGE_COUNT} "
            "for a stable COLMAP reconstruction."
        )

    for rec in records:
        if min(rec.width, rec.height) < MIN_RESOLUTION:
            report.rejected.append(str(rec.path))
            continue
        try:
            with Image.open(rec.path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.float64)
            if gray.shape[0] > 2 and gray.shape[1] > 2:
                variance = _laplacian_variance(gray)
                if variance < BLUR_VARIANCE_THRESHOLD:
                    report.warnings.append(f"Possibly blurry: {rec.path.name} (var={variance:.1f})")
        except Exception as exc:
            report.warnings.append(f"Could not analyze {rec.path.name}: {exc}")

    if report.rejected:
        report.passed = False

    return report
