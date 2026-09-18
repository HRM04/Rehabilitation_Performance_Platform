import csv


def create_interaction_logger(output_path):

    csv_file = open(
        output_path,
        "w",
        newline=""
    )

    fieldnames = [
        "session_id",
        "trial_id",
        "repetition_number",

        "frame_number",
        "timestamp_ms",

        "task_id",
        "task_name",

        "handedness",
        "hand_confidence",

        "hand_center_x",
        "hand_center_y",

        "grasp_detected",

        "interaction_state",
        "object_outcome",
        "fingertip_count",
        "object_distance",
        "object_movement",

        "object_class",
        "object_confidence",

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


def log_frame_record(
    writer,
    frame_record
):

    hand = frame_record["hand"]
    interaction = frame_record["interaction"]
    task = frame_record["task"]
    detected_object = frame_record["object"]
    trial = frame_record["trial"]

    hand_center_x, hand_center_y = hand["hand_center"]

    row = {
        "frame_number": frame_record["frame_number"],
        "timestamp_ms": frame_record["timestamp_ms"],

        "session_id": trial["session_id"],
        "trial_id": trial["trial_id"],
        "repetition_number": trial["repetition_number"],

        "task_id": task["task_id"],
        "task_name": task["task_name"],

        "handedness": hand["handedness"],
        "hand_confidence": hand["hand_confidence"],

        "hand_center_x": hand_center_x,
        "hand_center_y": hand_center_y,

        "grasp_detected": hand["grasp_detected"],

        "interaction_state": interaction["state"],
        "object_outcome": interaction["object_outcome"],
        "fingertip_count": interaction["fingertip_count"],
        "object_distance": interaction["object_distance"],
        "object_movement": interaction["object_movement"],

        "object_class": None,
        "object_confidence": None,

        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,

        "object_center_x": None,
        "object_center_y": None,
    }

    if detected_object is not None:

        x1, y1, x2, y2 = detected_object["bounding_box"]
        object_center_x, object_center_y = detected_object["center"]

        row.update({
            "object_class": detected_object["class_name"],
            "object_confidence": detected_object["confidence"],

            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,

            "object_center_x": object_center_x,
            "object_center_y": object_center_y,
        })

    writer.writerow(row)