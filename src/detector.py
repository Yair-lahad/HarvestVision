from ultralytics import YOLO

from config import DETECTION_MODEL


class FruitDetector:
    def __init__(self, model_name: str = DETECTION_MODEL):
        self.model = YOLO(model_name)

    def detect(self, image):
        return self.model(image, verbose=False)[0]
