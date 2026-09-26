import csv


POSE_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]


def create_pose_logger(output_path):
    """Create a CSV file for upper-body landmark measurements."""

    csv_file = open(output_path, "w", newline="")

    fieldnames = [
        "frame_number",
        "timestamp_ms",
        "landmark_name",
        "x_norm",
        "y_norm",
        "z_rel",
        "visibility",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    return csv_file, writer


def log_pose_frame(
    writer,
    frame_number,
    timestamp_ms,
    upper_body_landmarks,
    upper_body_angles=None,

):
    """Save one row per detected upper-body landmark."""

    if upper_body_landmarks is None:
        return

    if upper_body_angles is None:
        upper_body_angles = {}

    for landmark_name in POSE_LANDMARKS:
        landmark = upper_body_landmarks.get(landmark_name)

        if landmark is None:
            continue

        writer.writerow({
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "landmark_name": landmark_name,
            "x_norm": landmark["x"],
            "y_norm": landmark["y"],
            "z_rel": landmark["z"],
            "visibility": landmark["visibility"],
            "left_elbow_angle": upper_body_angles.get("left_elbow"),
            "right_elbow_angle": upper_body_angles.get("right_elbow"),
            "left_shoulder_angle": upper_body_angles.get("left_shoulder"),
            "right_shoulder_angle": upper_body_angles.get("right_shoulder"),
        })