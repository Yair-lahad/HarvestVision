from ultralytics import YOLO


class FruitDetector:
    def __init__(self, model_name: str = "yolo11x.pt"):
        self.model = YOLO(model_name)

    def detect(self, image):
        return self.model(image, verbose=False)[0]
