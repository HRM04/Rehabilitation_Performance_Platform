"""
Trajectory movement-quality assessment.

Uses timestamped object-centre observations from the trajectory game.

All spatial measurements are in image pixels, not physical units.
Technical measurements are not validated clinical scores.
"""

import math
import statistics


def assess_trajectory_movement_quality(
    samples,
    gap_threshold_ms=400,
    reversal_threshold_px=5,
):
    """Calculate basic movement and tracking-quality measurements."""

    result = {
        "assessment_available": False,
        "observation_count": len(samples),
        "average_speed_px_s": None,
        "peak_speed_px_s": None,
        "speed_variability_px_s": None,
        "movement_reversals": None,
        "tracking_id_changes": 0,
        "sampling_gap_count": 0,
        "largest_sampling_gap_ms": None,
        "median_sampling_interval_ms": None,
        "smoothness": {
            "available": False,
            "value": None,
            "reason": "Not yet implemented or insufficient data quality",
        },
        "quality_warnings": [],
    }

    if len(samples) < 3:
        result["quality_warnings"].append(
            "Insufficient trajectory observations"
        )
        return result

    intervals = []
    speeds = []
    directions = []
    id_changes = 0

    for previous, current in zip(samples, samples[1:]):
        dt = (
            current["timestamp_ms"]
            - previous["timestamp_ms"]
        )

        if dt <= 0:
            result["quality_warnings"].append(
                "Non-increasing trajectory timestamps"
            )
            continue

        x1, y1 = previous["center"]
        x2, y2 = current["center"]

        dx = x2 - x1
        dy = y2 - y1

        distance = math.hypot(dx, dy)
        speed = distance / (dt / 1000)

        intervals.append(dt)

        if (
            previous.get("object_id")
            != current.get("object_id")
        ):
            id_changes += 1
            continue

        if dt > gap_threshold_ms:
            continue

        speeds.append(speed)

        if abs(dx) >= reversal_threshold_px:
            directions.append(1 if dx > 0 else -1)

    result["tracking_id_changes"] = id_changes
    result["sampling_gap_count"] = sum(
        dt > gap_threshold_ms for dt in intervals
    )

    if intervals:
        result["largest_sampling_gap_ms"] = max(intervals)
        result["median_sampling_interval_ms"] = (
            statistics.median(intervals)
        )

    if id_changes:
        result["quality_warnings"].append(
            f"{id_changes} object tracking ID change(s)"
        )

    if result["sampling_gap_count"]:
        result["quality_warnings"].append(
            "Trajectory contains sampling gaps"
        )

    if not speeds:
        result["quality_warnings"].append(
            "No valid movement intervals"
        )
        return result

    result["assessment_available"] = True
    result["average_speed_px_s"] = statistics.mean(speeds)
    result["peak_speed_px_s"] = max(speeds)

    result["speed_variability_px_s"] = (
        statistics.pstdev(speeds)
        if len(speeds) > 1
        else 0.0
    )

    reversals = sum(
        previous != current
        for previous, current in zip(
            directions, directions[1:]
        )
    )

    result["movement_reversals"] = reversals

    return result
