"""Personalized reference-posture calibration for Version 1.

Uses multiple valid pose observations to estimate a reference
posture. Does not establish an ideal or clinically neutral posture.
"""

import statistics
import math


REQUIRED_LANDMARKS = ("left_shoulder", "right_shoulder")
OPTIONAL_LANDMARKS = ("left_hip", "right_hip")
POSTURE_LANDMARKS = REQUIRED_LANDMARKS + OPTIONAL_LANDMARKS


def landmark_visible(landmarks, name, threshold=0.5):
    point = landmarks.get(name)
    if not isinstance(point, dict):
        return False
    try:
        x, y = float(point["x"]), float(point["y"])
        visibility = float(point.get("visibility", 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(x) and math.isfinite(y)
        and math.isfinite(visibility)
        and visibility >= threshold
    )


def usable_frames(frames, threshold=0.5):
    result = []
    for frame in frames or []:
        if not isinstance(frame, dict):
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, dict):
            continue
        if all(
            landmark_visible(landmarks, name, threshold)
            for name in REQUIRED_LANDMARKS
        ):
            result.append(landmarks)
    return result


def calibration_subset(frames, minimum=10, threshold=0.5):
    shoulder_frames = usable_frames(frames, threshold)
    if len(shoulder_frames) < minimum:
        return None, None, None

    full_frames = [
        f for f in shoulder_frames
        if all(
            landmark_visible(f, name, threshold)
            for name in OPTIONAL_LANDMARKS
        )
    ]
    if len(full_frames) >= minimum:
        return full_frames, POSTURE_LANDMARKS, "FULL"

    return shoulder_frames, REQUIRED_LANDMARKS, "PARTIAL"


def calibrate_reference_posture(
    pose_frames, min_valid_frames=10, min_visibility=0.5
):
    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be at least 1")

    frames, names, mode = calibration_subset(
        pose_frames, min_valid_frames, min_visibility
    )
    if frames is None:
        return None

    reference = {}
    for name in names:
        reference[name] = {
            "x": statistics.median(float(f[name]["x"]) for f in frames),
            "y": statistics.median(float(f[name]["y"]) for f in frames),
            "visibility": 1.0,
        }

    full = mode == "FULL"
    return {
        "landmarks": reference,
        "valid_frames": len(frames),
        "method": "median_landmark_position",
        "reference_type": "observed_calibration_posture",
        "calibration_mode": mode,
        "trunk_reference_available": full,
        "shoulder_elevation_reference_available": full,
        "unavailable_measurements": [] if full else [
            "trunk_posture",
            "shoulder_elevation_relative_to_hips",
        ],
    }
