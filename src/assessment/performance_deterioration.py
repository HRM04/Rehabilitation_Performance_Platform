def percent_change(first_value, last_value):
    """
    Calculate percentage change from the first value
    to the last value.

    Returns None when either value is missing
    or when the first value is zero.
    """

    if first_value is None or last_value is None:
        return None

    if first_value == 0:
        return None

    return (
        (last_value - first_value)
        / abs(first_value)
    ) * 100.0


def calculate_performance_deterioration(
    repetition_results,
):
    """
    Compare early and late repetitions.

    This detects measurable performance change.
    It should not be interpreted as physiological fatigue
    without additional evidence.
    """

    valid_repetitions = [
        rep for rep in repetition_results
        if rep is not None
    ]

    if len(valid_repetitions) < 2:
        return None

    first_rep = valid_repetitions[0]
    last_rep = valid_repetitions[-1]

    result = {
        "duration_change_percent": percent_change(
            first_rep["duration_ms"],
            last_rep["duration_ms"],
        ),
        "joints": {},
        "trunk": {},
    }

    joint_names = [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
    ]

    metric_names = [
        "rom_deg",
        "mean_angular_speed_deg_s",
        "peak_angular_speed_deg_s",
        "mean_abs_angular_acceleration_deg_s2",
        "velocity_reversals",
    ]

    for joint_name in joint_names:

        result["joints"][joint_name] = {}

        for metric_name in metric_names:

            first_value = (
                first_rep["joints"][joint_name]
                .get(metric_name)
            )

            last_value = (
                last_rep["joints"][joint_name]
                .get(metric_name)
            )

            result["joints"][joint_name][metric_name] = {
                "first": first_value,
                "last": last_value,
                "change_percent": percent_change(
                    first_value,
                    last_value,
                ),
            }

    first_trunk = (
        first_rep["trunk"]
        .get("maximum_displacement_norm")
    )

    last_trunk = (
        last_rep["trunk"]
        .get("maximum_displacement_norm")
    )

    result["trunk"] = {
        "first": first_trunk,
        "last": last_trunk,
        "change_percent": percent_change(
            first_trunk,
            last_trunk,
        ),
    }

    return result