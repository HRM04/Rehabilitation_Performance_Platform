"""Temporal analysis of posture-related movement.

Describes threshold exceedances in measured posture signals.
Does not establish whether movement is clinically compensatory.
"""


def analyze_posture_deviations(
    samples,
    threshold,
    max_gap_ms=150,
):
    """
    Analyze timestamped posture measurements.

    samples: list of dictionaries containing:
        timestamp_ms: frame timestamp
        value: signed measurement, or None if unavailable

    threshold: positive threshold in the measurement's units
    max_gap_ms: maximum interval between consecutive samples
                considered part of the same episode

    Duration is estimated from consecutive valid sample intervals.
    """

    if threshold <= 0:
        raise ValueError("Threshold must be positive.")

    if max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive.")

    ordered = sorted(
        samples,
        key=lambda sample: sample["timestamp_ms"],
    )

    valid_count = 0
    exceeding_count = 0
    episodes = []
    active_start = None
    active_end = None
    previous_time = None
    previous_exceeding = False

    for sample in ordered:
        timestamp = sample["timestamp_ms"]
        value = sample.get("value")

        valid = value is not None
        exceeding = valid and abs(value) > threshold

        if valid:
            valid_count += 1
            if exceeding:
                exceeding_count += 1

        continuous = (
            previous_time is not None
            and timestamp > previous_time
            and timestamp - previous_time <= max_gap_ms
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

        previous_time = timestamp
        previous_exceeding = exceeding

    if active_start is not None:
        episodes.append({
            "start_ms": active_start,
            "end_ms": active_end,
            "duration_ms": active_end - active_start,
        })

    durations = [
        episode["duration_ms"]
        for episode in episodes
    ]

    return {
        "threshold": threshold,
        "valid_samples": valid_count,
        "exceeding_samples": exceeding_count,
        "exceedance_percent": (
            round(100 * exceeding_count / valid_count, 1)
            if valid_count else None
        ),
        "episode_count": len(episodes),
        "longest_episode_ms": (
            max(durations) if durations else 0
        ),
        "total_episode_duration_ms": sum(durations),
        "episodes": episodes,
        "duration_method": "observed_sample_span",
    }
