import csv


def create_object_logger(output_path):
    """
    Create a CSV logger for object detections.

    Each detected object produces one row per frame,
    independently of whether a hand is detected.
    """

    csv_file = open(
        output_path,
        "w",
        newline=""
    )

    fieldnames = [
        "session_id",
        "frame_number",
        "timestamp_ms",
        "task_id",
        "task_name",
        "object_class",
        "object_confidence",
        "track_id",
        "tracking_status",
        "object_id",
        "identity_status",
        "identity_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "object_center_x",
        "object_center_y",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    return csv_file, writer


def log_detected_objects(
    writer,
    detected_objects,
    session_id,
    frame_number,
    timestamp_ms,
    task_id,
    task_name,
):
    """
    Log every detected object in the current frame.

    Call this once per captured frame, including frames
    where no hands are detected.
    """

    for detected_object in detected_objects:

        x1, y1, x2, y2 = detected_object["bounding_box"]
        center_x, center_y = detected_object["center"]

        row = {
            "session_id": session_id,
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "task_id": task_id,
            "task_name": task_name,
            "object_class": detected_object["class_name"],
            "object_confidence": detected_object["confidence"],
            "track_id": detected_object.get("track_id"),
            "tracking_status": detected_object.get("tracking_status"),
            "object_id": detected_object.get("object_id"),
            "identity_status": detected_object.get("identity_status"),
            "identity_confidence": detected_object.get(
                "identity_confidence"
            ),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "object_center_x": center_x,
            "object_center_y": center_y,
        }

        writer.writerow(row)