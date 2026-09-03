"""Project-wide constants for the HarvestVision pipeline."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Recognised input formats
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")

# Annotated output formats
OUTPUT_IMAGE_SUFFIX = ".jpg"
OUTPUT_VIDEO_SUFFIX = ".mp4"
OUTPUT_VIDEO_CODEC = "mp4v"

# Detection model weights (Ultralytics will download on first use)
DETECTION_MODEL = "yolo11x.pt"

# Video window defaults (seconds)
DEFAULT_START_SEC = 0
DEFAULT_END_SEC = None
DEFAULT_INPUT_MODE = "video"


def first_file(directory: Path, suffixes: tuple[str, ...]) -> Path | None:
    """First file in a directory (sorted by name) whose suffix is in `suffixes`."""
    matches = sorted(
        p for p in directory.glob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )
    return matches[0] if matches else None


# First matching file in data/, or None if the folder has no such file
DEFAULT_IMAGE_PATH = first_file(DATA_DIR, IMAGE_SUFFIXES)
DEFAULT_VIDEO_PATH = first_file(DATA_DIR, VIDEO_SUFFIXES)