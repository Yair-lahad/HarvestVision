from __future__ import annotations

import numpy as np


class FruitTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.active: dict[int, np.ndarray] = {}
        self.ages: dict[int, int] = {}

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def update(self, boxes: list[tuple[float, float, float, float]]):
        """Match detections to tracks by IoU and keep only valid aged tracks."""
        next_active: dict[int, np.ndarray] = {}
        next_ages: dict[int, int] = {}
        assigned = set()

        for box in boxes:
            box_array = np.asarray(box, dtype=np.float32)
            best_id = None
            best_iou = 0.0

            for track_id in list(self.active):
                if track_id in assigned:
                    continue
                iou = self._iou(self.active[track_id], box_array)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            assigned.add(best_id)
            next_active[best_id] = box_array
            next_ages[best_id] = 0

        for track_id, prev_box in self.active.items():
            if track_id in assigned:
                continue
            age = self.ages.get(track_id, 0) + 1
            if age <= self.max_age:
                next_active[track_id] = prev_box
                next_ages[track_id] = age

        self.active = next_active
        self.ages = next_ages

        return [(track_id, tuple(box.tolist())) for track_id, box in self.active.items()]
