import cv2
import numpy as np
from ultralytics import YOLO


class ObjectSegmenter:
    def __init__(self, model_path="yolo11n-seg.pt", confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def segment_objects(self, frame, allowed_classes=None):
        """
        Segment objects in a BGR image.

        Returns one dictionary per detected object.
        The mask and contour describe the visible object outline.
        """
        class_ids = None

        if allowed_classes is not None:
            class_ids = [
                class_id
                for class_id, class_name in self.model.names.items()
                if class_name in allowed_classes
            ]

            if not class_ids:
                return []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=class_ids,
            verbose=False,
        )

        segmented_objects = []

        for result in results:
            if result.boxes is None or result.masks is None:
                continue

            for index, box in enumerate(result.boxes):
                if index >= len(result.masks.xy):
                    continue

                # Polygon coordinates are in the original image space.
                polygon = np.asarray(
                    result.masks.xy[index],
                    dtype=np.float32,
                )

                if len(polygon) < 3:
                    continue

                class_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Create a binary mask at the camera frame's resolution.
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                cv2.fillPoly(
                    mask,
                    [np.rint(polygon).astype(np.int32)],
                    255,
                )

                segmented_objects.append({
                    "class_name": self.model.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bounding_box": (x1, y1, x2, y2),
                    "center": (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                    ),
                    "polygon": polygon,
                    "mask": mask,
                })

        return segmented_objects