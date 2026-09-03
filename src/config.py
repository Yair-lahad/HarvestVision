"""Central configuration for the HarvestVision pipeline.

Everything that is a deliberate project choice (which classes count as
"fruit", default model, thresholds) lives here rather than on the CLI.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# COCO class IDs the pipeline treats as fruit, mapped to a display name.
# Hardcoded on purpose - the project counts fruit, not arbitrary objects.
FRUIT_CLASSES: dict[int, str] = {
    46: "banana",
    47: "apple",
    49: "orange",
}
FRUIT_CLASS_IDS: list[int] = sorted(FRUIT_CLASSES)

# Detection / tracking defaults (overridable via CLI flags).
DEFAULT_MODEL = "yolo11x.pt"
DEFAULT_CONF = 0.25
TRACKER_CONFIG = "bytetrack.yaml"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
