"""
Connect timestamped trajectory attempts to upper-body pose measurements.

Pose CSV contains one landmark per row.
This adapter groups landmarks by frame before running the existing
movement-quality calculations.
"""

import csv
import math
from collections import defaultdict

from src.assessment.repetition_metrics import (
    calculate_repetition_metrics,
)


ANGLE_COLUMNS = {
    "left_elbow": "left_elbow_angle",
    "right_elbow": "right_elbow_angle",
    "left_shoulder": "left_shoulder_angle",
    "right_shoulder": "right_shoulder_angle",
}


def optional_float(value):
    if value in (None, ""):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def load_pose_frames(csv_path):
    """Load and group pose CSV rows into timestamped frames."""

    frames = {}

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                frame_number = int(row["frame_number"])
                timestamp_ms = int(row["timestamp_ms"])
            except (ValueError, TypeError, KeyError):
                continue

            if frame_number < 0 or timestamp_ms < 0:
                continue

            if (
                frame_number in frames
                and frames[frame_number]["timestamp_ms"] != timestamp_ms
            ):
                # Conflicting timestamps cannot represent one frame.
                continue

            if frame_number not in frames:
                frames[frame_number] = {
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": {},
                    "angles": {
                        name: None for name in ANGLE_COLUMNS
                    },
                }

            frame = frames[frame_number]

            landmark_name = row.get("landmark_name")

            x = optional_float(row.get("x_norm"))
            y = optional_float(row.get("y_norm"))
            visibility = optional_float(row.get("visibility"))

            if (
                landmark_name
                and x is not None
                and y is not None
                and visibility is not None
            ):
                frame["landmarks"][landmark_name] = {
                    "x": x,
                    "y": y,
                    "visibility": visibility,
                }

            for name, column in ANGLE_COLUMNS.items():
                value = optional_float(row.get(column))

                if value is not None:
                    frame["angles"][name] = value

    return sorted(
        frames.values(),
        key=lambda frame: frame["timestamp_ms"],
    )


def assess_pose_for_attempts(session, pose_frames, reference_posture=None):
    """Align pose measurements with trajectory-game attempts."""

    # Exclude malformed frames without changing the source data.
    valid_frames = []

    for frame in pose_frames:
        if not isinstance(frame, dict):
            continue

        try:
            timestamp = float(frame["timestamp_ms"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if not math.isfinite(timestamp) or timestamp < 0:
            continue

        valid_frames.append({
            **frame,
            "timestamp_ms": timestamp,
            "angles": (
                frame.get("angles")
                if isinstance(frame.get("angles"), dict)
                else {}
            ),
            "landmarks": (
                frame.get("landmarks")
                if isinstance(frame.get("landmarks"), dict)
                else {}
            ),
        })

    valid_frames.sort(key=lambda frame: frame["timestamp_ms"])

    results = []

    for attempt in session.get("attempts", []):
        start = attempt.get("start_timestamp_ms")
        end = attempt.get("end_timestamp_ms")

        try:
            start = float(start)
            end = float(end)
            valid_window = (
                math.isfinite(start)
                and math.isfinite(end)
                and start >= 0
                and end >= start
            )
        except (TypeError, ValueError, OverflowError):
            valid_window = False

        if not valid_window:
            results.append({
                "attempt_number": attempt.get("attempt_number"),
                "movement_quality": None,
                "pose_quality": {
                    "status": "UNAVAILABLE",
                    "reason": "Missing or invalid attempt timestamps",
                },
            })
            continue

        matching = [
            frame for frame in valid_frames
            if start <= frame["timestamp_ms"] <= end
        ]

        movement = calculate_repetition_metrics(
            valid_frames,
            start,
            end,
            reference_posture=reference_posture,
        )

        angle_coverage = {}

        for joint in ANGLE_COLUMNS:
            count = sum(
                frame["angles"].get(joint) is not None
                for frame in matching
            )

            angle_coverage[joint] = {
                "valid_frames": count,
                "total_frames": len(matching),
                "coverage_percent": (
                    round(100 * count / len(matching), 1)
                    if matching else 0.0
                ),
            }

        trunk_frames = (
            movement.get("trunk", {}).get("valid_frames", 0)
            if movement else 0
        )

        results.append({
            "attempt_number": attempt.get("attempt_number"),
            "repetition_number": attempt.get("repetition_number"),
            "outcome": attempt.get("outcome"),
            "movement_quality": movement,
            "pose_quality": {
                "status": (
                    "PARTIAL" if movement else "UNAVAILABLE"
                ),
                "pose_frames": len(matching),
                "joint_angle_coverage": angle_coverage,
                "valid_trunk_frames": trunk_frames,
            },
        })

    return results
