"""Personalized reference-posture calibration for Version 1.

Uses multiple valid pose observations to estimate a reference
posture. Does not establish an ideal or clinically neutral posture.
"""

import statistics


POSTURE_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


def calibrate_reference_posture(
    pose_frames,
    min_valid_frames=10,
    min_visibility=0.5,
):
    """Estimate reference landmark positions from valid frames.

    Returns None when insufficient complete pose observations
    are available.
    """

    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be at least 1.")

    valid_landmarks = []

    for frame in pose_frames:
        landmarks = frame.get("landmarks") or {}

        if any(
            name not in landmarks
            or landmarks[name].get("visibility", 0) < min_visibility
            for name in POSTURE_LANDMARKS
        ):
            continue

        valid_landmarks.append(landmarks)

    if len(valid_landmarks) < min_valid_frames:
        return None

    reference = {}

    for name in POSTURE_LANDMARKS:
        reference[name] = {
            "x": statistics.median(
                frame[name]["x"]
                for frame in valid_landmarks
            ),
            "y": statistics.median(
                frame[name]["y"]
                for frame in valid_landmarks
            ),
            "visibility": 1.0,
        }

    return {
        "landmarks": reference,
        "valid_frames": len(valid_landmarks),
        "method": "median_landmark_position",
        "reference_type": "observed_calibration_posture",
    }
