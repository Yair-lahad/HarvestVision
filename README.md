# HarvestVision

HarvestVision is a small computer vision project for detecting and tracking fruit in orchard scenes.

The goal is to build a simple but clean pipeline that starts with raw input and ends with a fruit count.

## Pipeline

`image/video → detection → tracking → fruit count`

At a high level:

1. Input image or video is loaded.
2. A YOLO model detects fruit objects and returns bounding boxes, class IDs, and confidence scores.
3. A tracker keeps the same fruit identity across multiple frames.
4. A counter aggregates the tracked fruits into a final count.

## Current project structure

* `src/detector.py` — wraps Ultralytics YOLO for fruit detection
* `src/tracker.py` — links detections across frames by object position
* `src/counter.py` — tracks unique fruit IDs and total counts
* `src/main.py` — runs the detection pipeline on images or videos
* `data/` — hold raw data to analyze, gitignored
* `results/<timestamp>` — saves each run results: annotated and json helper, gitignored


## Why this matters

We first need to detect fruit correctly before we can count it. Once detection is reliable, tracking makes sure the same fruit is not counted multiple times across frames. The count is then based on unique tracked fruit identities instead of raw repeated detections.

## Stack

* Python
* Ultralytics YOLO based on PyTorch
* OpenCV

## Getting Started

Make sure Python is installed, then run:

```bash
pip install -r requirements.txt
```

```bash
python src/main.py --image data/sample_oranges.jpg
```

To process only part of a video, select a time window:

```bash
python src/main.py --video data/orchard_clip.webm --start-sec 7 --end-sec 30
```


