from __future__ import annotations

import argparse

from config import (
    DATA_DIR,
    DEFAULT_END_SEC,
    DEFAULT_IMAGE_PATH,
    DEFAULT_INPUT_MODE,
    DEFAULT_START_SEC,
    DEFAULT_VIDEO_PATH,
    RESULTS_DIR,
)
from pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit detection, tracking, and counting")
    parser.add_argument("--mode", choices=["image", "video"], default=DEFAULT_INPUT_MODE, help="Input mode")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Path to input image")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC, help="Video start time in seconds")
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC, help="Video end time in seconds")

    args = parser.parse_args()
    selected = args.video if args.mode == "video" else args.image
    if selected is None:
        parser.error(f"no {args.mode} file found in {DATA_DIR}; pass --{args.mode} explicitly")
    return args


def main():
    args = parse_args()
    run(
        args.mode,
        image_path=args.image,
        video_path=args.video,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        output_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
