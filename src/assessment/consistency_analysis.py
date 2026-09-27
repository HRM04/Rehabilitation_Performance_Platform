import statistics


def coefficient_of_variation(values):
    """
    Calculate coefficient of variation.

    Returns:
        Standard deviation / mean

    Lower values indicate more consistent measurements.

    Returns None when there are fewer than two valid values
    or when the mean is zero.
    """

    valid_values = [
        value for value in values
        if value is not None
    ]

    if len(valid_values) < 2:
        return None

    mean_value = statistics.mean(valid_values)

    if mean_value == 0:
        return None

    return (
        statistics.stdev(valid_values)
        / abs(mean_value)
    )


def calculate_repetition_consistency(
    repetition_results,
):
    """
    Compare movement metrics across repetitions.

    repetition_results should contain the output from
    calculate_repetition_metrics() for multiple repetitions.
    """

    if len(repetition_results) < 2:
        return None

    consistency = {
        "duration": {},
        "joints": {},
        "trunk": {},
    }

    durations = [
        rep.get("duration_ms")
        for rep in repetition_results
    ]

    consistency["duration"] = {
        "mean_ms": statistics.mean(durations),
        "cv": coefficient_of_variation(durations),
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

        consistency["joints"][joint_name] = {}

        for metric_name in metric_names:

            values = [
                rep["joints"][joint_name].get(metric_name)
                for rep in repetition_results
                if rep is not None
            ]

            valid_values = [
                value for value in values
                if value is not None
            ]

            consistency["joints"][joint_name][metric_name] = {
                "mean": (
                    statistics.mean(valid_values)
                    if valid_values
                    else None
                ),
                "cv": coefficient_of_variation(values),
            }

    trunk_values = [
        rep["trunk"].get("maximum_displacement_norm")
        for rep in repetition_results
        if rep is not None
    ]

    valid_trunk_values = [
        value for value in trunk_values
        if value is not None
    ]

    consistency["trunk"] = {
        "mean_max_displacement": (
            statistics.mean(valid_trunk_values)
            if valid_trunk_values
            else None
        ),
        "cv": coefficient_of_variation(trunk_values),
    }

    return consistency