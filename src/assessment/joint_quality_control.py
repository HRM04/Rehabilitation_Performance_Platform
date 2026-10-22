"""
Quality screening for upper-body joint-angle measurements.

Thresholds are exploratory engineering rules, not clinical
validation criteria. Raw measurements are never modified.
"""

import math


def assess_joint_quality(
    frames,
    joint_name,
    min_coverage_percent=80.0,
    min_observations=10,
    jump_threshold_deg=30.0,
    speed_threshold_deg_s=300.0,
):
    observations = []

    for frame in frames:
        angle = frame.get("angles", {}).get(joint_name)
        timestamp = frame.get("timestamp_ms")

        if (
            isinstance(angle, (int, float))
            and not isinstance(angle, bool)
            and math.isfinite(angle)
            and isinstance(timestamp, (int, float))
            and math.isfinite(timestamp)
        ):
            observations.append((timestamp, angle))

    total_frames = len(frames)
    coverage = (
        100 * len(observations) / total_frames
        if total_frames else 0.0
    )

    warnings = []
    flagged_intervals = []

    if len(observations) < min_observations:
        warnings.append("Insufficient valid observations")

    if coverage < min_coverage_percent:
        warnings.append("Insufficient angle coverage")

    for (t1, a1), (t2, a2) in zip(
        observations, observations[1:]
    ):
        dt = t2 - t1

        if dt <= 0:
            continue

        change = abs(a2 - a1)
        speed = change / (dt / 1000)

        if (
            change > jump_threshold_deg
            or speed > speed_threshold_deg_s
        ):
            flagged_intervals.append({
                "start_timestamp_ms": t1,
                "end_timestamp_ms": t2,
                "angle_change_deg": round(change, 2),
                "angular_speed_deg_s": round(speed, 2),
            })

    if flagged_intervals:
        warnings.append(
            "Abrupt angle transitions require review"
        )

    reliable = not warnings

    return {
        "joint": joint_name,
        "status": "SUFFICIENT" if reliable else "REVIEW",
        "valid_observations": len(observations),
        "total_frames": total_frames,
        "coverage_percent": round(coverage, 1),
        "flagged_transition_count": len(flagged_intervals),
        "flagged_intervals": flagged_intervals,
        "warnings": warnings,
        "clinician_display_allowed": reliable,
    }
