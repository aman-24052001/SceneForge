"""
Image Fetcher Module
Responsibilities:
- Accept a local folder path of images (free, no API)
- Validate that images exist and are readable
- Return a normalized list of image paths + basic metadata (size, format)

No external API calls -- intentionally free / offline-first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int


class FetchError(Exception):
    """Raised when the input image folder is missing or has no usable images."""


def fetch_from_folder(folder: str | Path) -> list[ImageRecord]:
    """
    Scan a folder for supported images and return their metadata.

    Args:
        folder: path to a directory containing images.

    Returns:
        List of ImageRecord, sorted by filename for deterministic ordering.

    Raises:
        FetchError: if the folder doesn't exist or contains no valid images.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FetchError(f"Image folder not found: {folder}")

    records: list[ImageRecord] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            with Image.open(path) as img:
                records.append(ImageRecord(path=path, width=img.width, height=img.height))
        except Exception:
            continue  # skip unreadable/corrupt files

    if not records:
        raise FetchError(f"No supported images found in: {folder}")

    return records
