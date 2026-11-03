"""Additional engineering reliability measurements for object trajectories.

Does not modify legacy metrics or clinical assessment outcomes.
All distances are in image pixels.
"""
import math


def _progress_on_path(point, path):
    """Nearest-point arc-length coordinate on a reference polyline."""
    if len(path) < 2:
        return None

    px, py = point
    best_distance = float("inf")
    best_progress = None
    accumulated = 0.0

    for a, b in zip(path, path[1:]):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            continue

        fraction = max(
            0.0,
            min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq)
        )

        qx = ax + fraction * dx
        qy = ay + fraction * dy
        distance = math.hypot(px - qx, py - qy)
        segment_length = math.sqrt(length_sq)
        progress = accumulated + fraction * segment_length

        if distance < best_distance:
            best_distance = distance
            best_progress = progress

        accumulated += segment_length

    return best_progress


def assess_trajectory_reliability(
    samples,
    desired_path,
    gap_threshold_ms=400,
    reversal_threshold_px=5,
    minimum_samples=5,
    minimum_temporal_coverage=0.80,
):
    """Return supplemental measurements and their engineering eligibility."""

    result = {
        "sample_count": len(samples),
        "valid_interval_count": 0,
        "total_interval_count": max(0, len(samples) - 1),
        "valid_observed_duration_ms": 0,
        "observation_span_ms": None,
        "temporal_coverage": None,
        "time_weighted_average_speed_px_s": None,
        "path_relative_reversals": None,
        "tracking_id_changes": 0,
        "sampling_gap_count": 0,
        "invalid_timestamp_count": 0,
        "eligible": False,
        "reasons": [],
    }

    if len(samples) < 2:
        result["reasons"].append("Fewer than two observations")
        return result

    total_distance = 0.0
    valid_duration = 0
    reversals = 0
    previous_direction = None

    first_time = samples[0]["timestamp_ms"]
    last_time = samples[-1]["timestamp_ms"]

    if not all(
        isinstance(t, (int, float))
        and not isinstance(t, bool)
        and math.isfinite(t)
        for t in (first_time, last_time)
    ):
        result["reasons"].append("Invalid observation timestamps")
        return result

    span = last_time - first_time
    result["observation_span_ms"] = span

    if span <= 0:
        result["reasons"].append("Non-positive observation duration")
        return result

    progress = [
        _progress_on_path(s["center"], desired_path)
        for s in samples
    ]

    for i, (previous, current) in enumerate(zip(samples, samples[1:])):
        t1 = previous["timestamp_ms"]
        t2 = current["timestamp_ms"]

        if not all(
            isinstance(t, (int, float))
            and not isinstance(t, bool)
            and math.isfinite(t)
            for t in (t1, t2)
        ):
            result["invalid_timestamp_count"] += 1
            previous_direction = None
            continue

        dt = t2 - t1

        if dt <= 0:
            result["invalid_timestamp_count"] += 1
            previous_direction = None
            continue

        id_changed = (
            previous.get("object_id") != current.get("object_id")
        )
        sampling_gap = dt > gap_threshold_ms

        # These diagnostics are independent: one interval can
        # contain both an identity change and a sampling gap.
        if id_changed:
            result["tracking_id_changes"] += 1

        if sampling_gap:
            result["sampling_gap_count"] += 1

        if id_changed or sampling_gap:
            previous_direction = None
            continue

        x1, y1 = previous["center"]
        x2, y2 = current["center"]

        distance = math.hypot(x2 - x1, y2 - y1)
        if not math.isfinite(distance):
            result["reasons"].append("Non-finite object position")
            previous_direction = None
            continue

        total_distance += distance
        valid_duration += dt
        result["valid_interval_count"] += 1

        p1, p2 = progress[i], progress[i + 1]
        if p1 is None or p2 is None:
            previous_direction = None
            continue

        delta = p2 - p1

        if abs(delta) >= reversal_threshold_px:
            direction = 1 if delta > 0 else -1

            if previous_direction is not None and direction != previous_direction:
                reversals += 1

            previous_direction = direction

    result["valid_observed_duration_ms"] = valid_duration
    result["temporal_coverage"] = valid_duration / span
    result["path_relative_reversals"] = (
        reversals if len(desired_path) >= 2 else None
    )

    if valid_duration > 0:
        result["time_weighted_average_speed_px_s"] = (
            total_distance / (valid_duration / 1000.0)
        )

    if len(samples) < minimum_samples:
        result["reasons"].append("Insufficient sample count")

    if result["temporal_coverage"] < minimum_temporal_coverage:
        result["reasons"].append("Insufficient temporal coverage")

    if result["tracking_id_changes"]:
        result["reasons"].append("Tracking identity changed")

    if result["invalid_timestamp_count"]:
        result["reasons"].append("Invalid or non-increasing timestamps")

    if valid_duration <= 0:
        result["reasons"].append("No valid movement intervals")

    if result["path_relative_reversals"] is None:
        result["reasons"].append("Reference path unavailable")

    result["eligible"] = not result["reasons"]
    return result
