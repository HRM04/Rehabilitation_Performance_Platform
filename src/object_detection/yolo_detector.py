from ultralytics import YOLO


# Load pretrained YOLO model once
import os
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolo11m.pt")
model = YOLO(MODEL_NAME)
print(f"[DETECTOR] model: {MODEL_NAME}")


def detect_objects(frame):
    """
    Run YOLO object detection on one camera frame.

    Returns a list of detected objects.
    """

    # results = model(
    #     frame,
    #     verbose=False
    # )

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    detected_objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            class_name = model.names[class_id]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            detected_objects.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "center": (
                        center_x,
                        center_y
                    )
                }
            )

    return detected_objects



import cv2


def draw_detected_objects(
    frame,
    detected_objects
):
    """
    Draw YOLO detections onto the camera frame.
    """

    for detected_object in detected_objects:

        class_name = (
            detected_object["class_name"]
        )

        confidence = (
            detected_object["confidence"]
        )

        x1, y1, x2, y2 = (
            detected_object["bounding_box"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"{class_name}: "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame




def filter_objects_by_class(
    detected_objects,
    allowed_classes
):
    permitted_objects = []
    irrelevant_objects = []

    for detected_object in detected_objects:

        class_name = detected_object["class_name"]

        if class_name in allowed_classes:
            permitted_objects.append(detected_object)

        else:
            irrelevant_objects.append(detected_object)

    return permitted_objects, irrelevant_objects