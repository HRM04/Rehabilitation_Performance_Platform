def calculate_relative_change(
    baseline_value,
    current_value,
):
    """
    Calculate percentage change relative to a patient baseline.

    Returns None when either value is unavailable
    or when the baseline value is zero.
    """

    if baseline_value is None or current_value is None:
        return None

    if baseline_value == 0:
        return None

    return (
        (current_value - baseline_value)
        / abs(baseline_value)
    ) * 100.0


def compare_repetition_to_baseline(
    repetition_assessment,
    baseline,
):
    """
    Compare one rehabilitation repetition against the
    patient's own baseline performance.
    """

    if repetition_assessment is None or baseline is None:
        return None

    comparison = {
        "duration_change_percent": None,
        "joints": {},
        "compensation": {},
    }

    # -----------------------------------------
    # Duration comparison
    # -----------------------------------------

    current_duration = (
        repetition_assessment
        .get("timing", {})
        .get("duration_ms")
    )

    comparison["duration_change_percent"] = (
        calculate_relative_change(
            baseline.get("duration_ms"),
            current_duration,
        )
    )

    # -----------------------------------------
    # Joint comparisons
    # -----------------------------------------

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

    movement_quality = (
        repetition_assessment
        .get("movement_quality")
        or {}
    )

    for joint_name in joint_names:

        comparison["joints"][joint_name] = {}

        current_joint = (
            movement_quality
            .get("joints", {})
            .get(joint_name, {})
        )

        baseline_joint = (
            baseline
            .get("joints", {})
            .get(joint_name, {})
        )

        for metric_name in metric_names:

            comparison["joints"][joint_name][
                metric_name
            ] = calculate_relative_change(
                baseline_joint.get(metric_name),
                current_joint.get(metric_name),
            )

    # -----------------------------------------
    # Compensation comparison
    # -----------------------------------------

    current_trunk = (
        repetition_assessment
        .get("compensation", {})
        .get("maximum_trunk_displacement_norm")
        if repetition_assessment.get("compensation")
        is not None
        else None
    )

    baseline_trunk = (
        baseline
        .get("compensation", {})
        .get("maximum_trunk_displacement_norm")
    )

    comparison["compensation"][
        "trunk_displacement_change_percent"
    ] = calculate_relative_change(
        baseline_trunk,
        current_trunk,
    )

    return comparison