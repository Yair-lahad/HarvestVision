from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import cv2

from counter import FruitCounter
from detector import FruitDetector
from tracker import FruitTracker

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_MODE = "video"
DEFAULT_IMAGE_PATH = BASE_DIR / "data" / "sample_oranges.jpg"
DEFAULT_VIDEO_PATH = next(
    (path for path in sorted((BASE_DIR / "data").glob("*.webm")) if path.is_file()),
    BASE_DIR / "data" / "orchard_clip.webm",
)
DEFAULT_START_SEC = 7.0
DEFAULT_END_SEC = 30.0


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate

    if candidate.exists():
        return candidate

    data_dir = BASE_DIR / "data"
    matches = sorted(data_dir.glob("*.webm"))
    if matches:
        return matches[0]
    return candidate


def create_run_dir(output_dir: Path, input_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / input_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_summary(run_dir: Path, input_name: str, summary: dict) -> Path:
    summary_path = run_dir / f"summary_{input_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_image(image_path: str | Path, output_dir: str | Path | None = None):
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir is not None else BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    result = FruitDetector().detect(image)
    run_dir = create_run_dir(output_dir, image_path.stem)

    annotated_path = run_dir / f"annotated_{image_path.stem}.jpg"
    cv2.imwrite(str(annotated_path), result.plot())

    summary = {
        "input_name": image_path.stem,
        "source": str(image_path),
        "detections": len(result.boxes),
        "tracked": len(result.boxes),
        "unique_fruit_count": len(result.boxes),
        "output_dir": str(run_dir),
        "annotated_image": str(annotated_path),
    }

    write_summary(run_dir, image_path.stem, summary)
    return summary


def run_video(video_path: str | Path, start_sec: float = 0.0, end_sec: float | None = None, output_dir: str | Path | None = None):
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("start_sec must be a finite, non-negative number")
    if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
        raise ValueError("end_sec must be greater than start_sec")

    video_path = resolve_input_path(video_path)
    output_dir = Path(output_dir) if output_dir is not None else BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end_sec is None:
        end_sec = total_frames / fps if total_frames else 0.0

    start_frame = math.ceil(start_sec * fps)
    end_frame = math.ceil(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    run_dir = create_run_dir(output_dir, video_path.stem)

    annotated_path = run_dir / f"annotated_{video_path.stem}_{start_sec}s_{end_sec}.mp4"
    writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = FruitDetector()
    tracker = FruitTracker()
    counter = FruitCounter()
    frame_index = start_frame
    processed_frames = 0

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.detect(frame)
        boxes = []
        if len(result.boxes):
            boxes = [tuple(float(v) for v in box) for box in result.boxes.xyxy.cpu().tolist()]

        tracked = tracker.update(boxes)
        counter.update(tracked)
        processed_frames += 1

        if processed_frames % 10 == 0:
            print(
                f"Processed {processed_frames}/{end_frame - start_frame} frames",
                flush=True,
            )

        for track_id, (x1, y1, x2, y2) in tracked:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    unique_count = counter.total_count()
    summary = {
        "input_name": video_path.stem,
        "source": str(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frames_processed": processed_frames,
        "unique_fruit_count": unique_count,
        "output_dir": str(run_dir),
        "annotated_video": str(annotated_path),
    }

    write_summary(run_dir, video_path.stem, summary)
    return summary
