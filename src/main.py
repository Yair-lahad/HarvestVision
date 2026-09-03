from __future__ import annotations

import argparse
import sys

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
    parser.add_argument("--mode", choices=["image", "video"], default=None,
                         help="Input mode (inferred from --image/--video if omitted)")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC, help="Video start time in seconds")
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC, help="Video end time in seconds")

    args = parser.parse_args()

    if args.mode is None:
        tags = {"--image": "image", "--video": "video"}
        first_tag = next((tags[a] for a in sys.argv[1:] if a in tags), None)
        args.mode = first_tag or DEFAULT_INPUT_MODE

    args.image = args.image or DEFAULT_IMAGE_PATH
    args.video = args.video or DEFAULT_VIDEO_PATH

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
