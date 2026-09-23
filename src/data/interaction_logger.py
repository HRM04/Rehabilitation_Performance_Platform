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

        "track_id": None,
        "tracking_status": None,

        "object_id": None,
        "identity_status": None,
        "identity_confidence": None,

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

            "track_id": detected_object.get("track_id"),
            "tracking_status": detected_object.get("tracking_status"),

            "object_id": detected_object.get("object_id"),
"identity_status": detected_object.get("identity_status"),
"identity_confidence": detected_object.get("identity_confidence"),
        })

    writer.writerow(row)




def log_no_hand_interaction(
    writer,
    frame_number,
    timestamp_ms,
    session_id,
    trial_id,
    repetition_number,
    task_id,
    task_name,
    object_outcome
):
    """
    Record a frame where no hands were detected.

    Hand-specific and interaction measurements remain
    empty because no valid hand observation is available.
    """

    row = {
        "session_id": session_id,
        "trial_id": trial_id,
        "repetition_number": repetition_number,

        "frame_number": frame_number,
        "timestamp_ms": timestamp_ms,

        "task_id": task_id,
        "task_name": task_name,

        "handedness": None,
        "hand_confidence": None,

        "hand_center_x": None,
        "hand_center_y": None,

        "grasp_detected": None,

        "interaction_state": None,
        "object_outcome": object_outcome,
        "fingertip_count": None,
        "object_distance": None,
        "object_movement": None,

        "object_class": None,
        "object_confidence": None,

        "track_id": None,
        "tracking_status": None,

        "object_id": None,
        "identity_status": None,
        "identity_confidence": None,

        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,

        "object_center_x": None,
        "object_center_y": None
    }

    writer.writerow(row)