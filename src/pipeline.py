from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import cv2

import config
from counter import FruitCounter
from detector import FruitDetector
from tracker import FruitTracker


# --- pipeline ---------------------------------------------------------------

def run(mode: str, image_path, video_path, start_sec, end_sec, output_dir=None) -> dict:
    """Entry point: run the image or video pipeline and return its summary."""
    if mode == "video":
        return run_video(video_path, start_sec=start_sec, end_sec=end_sec, output_dir=output_dir)
    return run_image(image_path, output_dir=output_dir)


def run_image(image_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """Detect fruit in one image, then write an annotated copy and a summary."""
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    result = FruitDetector().detect(image)

    run_dir = create_run_dir(output_dir, image_path.stem)
    annotated_path = run_dir / f"annotated_{image_path.stem}{config.OUTPUT_IMAGE_SUFFIX}"
    cv2.imwrite(str(annotated_path), result.plot())

    summary = {
        "input_name": image_path.stem,
        "source": str(image_path),
        "unique_fruit_count": len(result.boxes),
        "output_dir": str(run_dir),
        "annotated_image": str(annotated_path),
    }
    write_summary(run_dir, image_path.stem, summary)
    return summary


def run_video(
    video_path: str | Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Detect, track and count fruit across a [start_sec, end_sec] window of the video."""
    validate_window(start_sec, end_sec)
    video_path = resolve_input_path(video_path)

    cap, fps, size, total_frames = open_video(video_path)
    if end_sec is None:
        end_sec = total_frames / fps if total_frames else 0.0
    start_frame = math.ceil(start_sec * fps)
    frame_count = math.ceil(end_sec * fps) - start_frame

    run_dir = create_run_dir(output_dir, video_path.stem)
    annotated_path = run_dir / f"annotated_{video_path.stem}_{start_sec}s_{end_sec}s{config.OUTPUT_VIDEO_SUFFIX}"
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, size)

    frames_processed, unique_count = track_window(cap, writer, start_frame, frame_count)
    cap.release()
    writer.release()

    summary = {
        "input_name": video_path.stem,
        "source": str(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frames_processed": frames_processed,
        "unique_fruit_count": unique_count,
        "output_dir": str(run_dir),
        "annotated_video": str(annotated_path),
    }
    write_summary(run_dir, video_path.stem, summary)
    return summary


# --- helpers -------------------------------------------------------------------

def resolve_input_path(path: str | Path) -> Path:
    """Return the path as given, or fall back to the first video in data/ if it is missing."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = config.BASE_DIR / candidate
    if candidate.exists():
        return candidate

    return config.first_file(config.DATA_DIR, config.VIDEO_SUFFIXES) or candidate


def validate_window(start_sec: float, end_sec: float | None) -> None:
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("start_sec must be a finite, non-negative number")
    if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
        raise ValueError("end_sec must be greater than start_sec")


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, tuple[int, int], int]:
    """Open a video and read its fps, (width, height) and frame count."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, size, total_frames


def create_run_dir(output_dir: str | Path | None, input_name: str) -> Path:
    base = Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / input_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(run_dir: Path, input_name: str, summary: dict) -> Path:
    summary_path = run_dir / f"summary_{input_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def draw_tracks(frame, tracked) -> None:
    for track_id, (x1, y1, x2, y2) in tracked:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def track_window(cap, writer, start_frame: int, frame_count: int) -> tuple[int, int]:
    """Run detect -> track -> count over the frame window; return (frames_processed, unique_count)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # not frame-accurate on every codec

    detector = FruitDetector()
    tracker = FruitTracker()
    counter = FruitCounter()
    processed = 0

    while processed < frame_count:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.detect(frame)
        boxes = [tuple(box) for box in result.boxes.xyxy.cpu().tolist()]
        tracked = tracker.update(boxes)
        counter.update(tracked)
        processed += 1

        if processed % 10 == 0:
            print(f"Processed {processed}/{frame_count} frames", flush=True)

        draw_tracks(frame, tracked)
        writer.write(frame)

    return processed, counter.total_count()