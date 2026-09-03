from __future__ import annotations

import argparse

from pipeline import (
    BASE_DIR,
    DEFAULT_END_SEC,
    DEFAULT_IMAGE_PATH,
    DEFAULT_INPUT_MODE,
    DEFAULT_START_SEC,
    DEFAULT_VIDEO_PATH,
    run_image,
    run_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit detection, tracking, and counting")
    parser.add_argument("--mode", choices=["image", "video"], default=DEFAULT_INPUT_MODE, help="Input mode")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE_PATH), help="Path to input image")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO_PATH), help="Path to input video")
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC, help="Video start time in seconds")
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC, help="Video end time in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = BASE_DIR / "results"

    if args.mode == "video":
        run_video(args.video, start_sec=args.start_sec, end_sec=args.end_sec, output_dir=output_dir)
    else:
        run_image(args.image, output_dir=output_dir)


if __name__ == "__main__":
    main()
