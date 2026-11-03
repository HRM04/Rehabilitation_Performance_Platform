"""Temporal analysis of image-based posture deviations.

Threshold exceedances are engineering indicators, not validated
clinical compensation classifications.
"""

import math


def analyze_posture_deviations(
    samples,
    threshold,
    max_gap_ms=250,
    min_episode_ms=300,
):
    """Analyze timestamped posture deviations and data availability."""

    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be positive and finite.")

    if not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive and finite.")

    if not math.isfinite(min_episode_ms) or min_episode_ms < 0:
        raise ValueError("min_episode_ms must be nonnegative and finite.")

    ordered = []

    for sample in samples:
        try:
            timestamp = float(sample["timestamp_ms"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if not math.isfinite(timestamp) or timestamp < 0:
            continue

        value = sample.get("value")

        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError, OverflowError):
            value = None

        if value is not None and not math.isfinite(value):
            value = None

        ordered.append((timestamp, value))

    ordered.sort(key=lambda item: item[0])

    episodes = []
    active_start = None
    active_end = None
    previous_timestamp = None
    previous_exceeding = False

    valid_count = 0
    exceeding_count = 0

    for timestamp, value in ordered:
        valid = value is not None
        exceeding = valid and abs(value) > threshold

        if valid:
            valid_count += 1
            if exceeding:
                exceeding_count += 1

        continuous = (
            previous_timestamp is not None
            and 0 < timestamp - previous_timestamp <= max_gap_ms
        )

        if exceeding:
            if active_start is None or not (
                previous_exceeding and continuous
            ):
                if active_start is not None:
                    episodes.append({
                        "start_ms": active_start,
                        "end_ms": active_end,
                        "duration_ms": active_end - active_start,
                    })

                active_start = timestamp

            active_end = timestamp

        elif active_start is not None:
            episodes.append({
                "start_ms": active_start,
                "end_ms": active_end,
                "duration_ms": active_end - active_start,
            })
            active_start = None
            active_end = None

        previous_timestamp = timestamp
        previous_exceeding = exceeding

    if active_start is not None:
        episodes.append({
            "start_ms": active_start,
            "end_ms": active_end,
            "duration_ms": active_end - active_start,
        })

    sustained = [
        episode
        for episode in episodes
        if episode["duration_ms"] >= min_episode_ms
    ]

    durations = [
        episode["duration_ms"]
        for episode in episodes
    ]

    total_samples = len(ordered)

    return {
        "threshold": threshold,
        "valid_samples": valid_count,
        "missing_samples": total_samples - valid_count,
        "total_samples": total_samples,
        "coverage_percent": (
            round(100 * valid_count / total_samples, 1)
            if total_samples else None
        ),
        "exceeding_samples": exceeding_count,
        "exceedance_percent": (
            round(100 * exceeding_count / valid_count, 1)
            if valid_count else None
        ),
        "episode_count": len(episodes),
        "longest_episode_ms": max(durations) if durations else 0,
        "total_episode_duration_ms": sum(durations),
        "episodes": episodes,
        "sustained_episode_count": len(sustained),
        "sustained_episodes": sustained,
        "min_episode_ms": min_episode_ms,
        "max_gap_ms": max_gap_ms,
        "duration_method": "observed_sample_span",
    }
