"""
Image Validator Module
Responsibilities:
- PRE-INPUT checks (before COLMAP runs at all):
    - Check image count meets minimum threshold for SfM
    - Check minimum resolution
    - Flag (not silently drop) low-quality images via blur heuristic
- POST-MATCHING checks (after COLMAP's feature_extractor + matcher run,
  before the expensive mapper/reconstruction stage):
    - Inspect database.db directly for verified geometric inlier counts.
    - This catches degenerate inputs early -- e.g. flat/texture-repetitive
      objects where SIFT matches look fine in raw count but RANSAC rejects
      almost all of them as geometrically inconsistent. This exact failure
      mode was hit during development (see docs/test_log.md): a flat-faced
      cube with checkerboard texture produced 0-16 inliers per pair instead
      of the 200+ a well-conditioned scene produces, and COLMAP's mapper
      silently gave up rather than erroring loudly.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from sceneforge.fetcher import ImageRecord

MIN_IMAGE_COUNT = 10
MIN_RESOLUTION = 200  # px, shorter side
BLUR_VARIANCE_THRESHOLD = 50.0  # Laplacian variance; lower = blurrier

# Post-matching thresholds, calibrated against what was observed in testing:
# degenerate (flat-texture) scenes topped out at ~16 inliers on their best
# pairs; a well-conditioned scene produced 200+ on its best pairs.
MIN_INLIERS_GOOD_PAIR = 50          # a pair needs at least this many inliers to count as "good"
MIN_GOOD_PAIRS = 1                  # need at least this many good pairs to proceed
MIN_PAIRS_WITH_ANY_INLIERS_RATIO = 0.3  # warn if fewer than 30% of pairs have any verified match


@dataclass
class ValidationReport:
    passed: bool
    image_count: int
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MatchQualityReport:
    passed: bool
    total_pairs: int
    pairs_with_inliers: int
    best_pair_inliers: int
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
    """Pre-input checks, run before COLMAP touches anything."""
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


def validate_match_quality(database_path: str | Path) -> MatchQualityReport:
    """
    Post-matching check: inspect COLMAP's database.db for verified geometric
    inlier counts (the `two_view_geometries` table), and fail loudly BEFORE
    the expensive mapper stage runs on a doomed reconstruction.

    Call this after colmap_runner has run feature_extractor + exhaustive_matcher
    but before running mapper.

    Args:
        database_path: path to COLMAP's database.db.

    Returns:
        MatchQualityReport. `passed=False` means the mapper stage is very
        likely to fail or produce a degenerate/empty reconstruction --
        the caller should stop and surface the warnings rather than burn
        CPU time on `mapper`.
    """
    database_path = Path(database_path)
    if not database_path.exists():
        return MatchQualityReport(
            passed=False, total_pairs=0, pairs_with_inliers=0, best_pair_inliers=0,
            warnings=[f"database.db not found at {database_path} -- did matching run?"],
        )

    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT rows FROM two_view_geometries")
        inlier_counts = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    total_pairs = len(inlier_counts)
    pairs_with_inliers = sum(1 for n in inlier_counts if n > 0)
    best_pair_inliers = max(inlier_counts) if inlier_counts else 0
    good_pairs = sum(1 for n in inlier_counts if n >= MIN_INLIERS_GOOD_PAIR)

    warnings: list[str] = []
    passed = True

    if total_pairs == 0:
        passed = False
        warnings.append("No image pairs found in database -- matching may not have run.")
    else:
        ratio_with_any = pairs_with_inliers / total_pairs
        if ratio_with_any < MIN_PAIRS_WITH_ANY_INLIERS_RATIO:
            warnings.append(
                f"Only {pairs_with_inliers}/{total_pairs} pairs "
                f"({ratio_with_any:.0%}) have any verified matches. This usually means "
                "insufficient image overlap, or a texture-repetitive scene causing "
                "RANSAC to reject most matches."
            )

        if good_pairs < MIN_GOOD_PAIRS:
            passed = False
            warnings.append(
                f"No pair reached {MIN_INLIERS_GOOD_PAIR}+ verified inliers "
                f"(best pair: {best_pair_inliers}). COLMAP's mapper is very likely to "
                "fail to find a good initial image pair. Common causes: too little "
                "camera motion between shots, flat/textureless surfaces, or "
                "repeating patterns (e.g. checkerboards, tiled floors) that cause "
                "SIFT mismatches RANSAC then rejects. See docs/test_log.md for a "
                "real example of this failure mode."
            )

    return MatchQualityReport(
        passed=passed,
        total_pairs=total_pairs,
        pairs_with_inliers=pairs_with_inliers,
        best_pair_inliers=best_pair_inliers,
        warnings=warnings,
    )
