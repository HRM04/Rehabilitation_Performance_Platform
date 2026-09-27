import statistics


def create_patient_baseline(
    repetition_assessments,
):
    """
    Create a preliminary patient-specific baseline from
    previously completed repetition assessments.

    The baseline uses the patient's own earlier performance
    rather than a universal threshold.
    """

    valid_repetitions = [
        rep
        for rep in repetition_assessments
        if rep is not None
    ]

    if not valid_repetitions:
        return None

    baseline = {
        "repetition_count": len(valid_repetitions),
        "duration_ms": None,
        "joints": {},
        "compensation": {},
    }

    # -----------------------------------------
    # Duration baseline
    # -----------------------------------------

    durations = [
        rep["timing"].get("duration_ms")
        for rep in valid_repetitions
        if rep.get("timing") is not None
        and rep["timing"].get("duration_ms") is not None
    ]

    if durations:
        baseline["duration_ms"] = statistics.mean(
            durations
        )

    # -----------------------------------------
    # Joint movement baselines
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

    for joint_name in joint_names:

        baseline["joints"][joint_name] = {}

        for metric_name in metric_names:

            values = []

            for rep in valid_repetitions:

                movement = rep.get(
                    "movement_quality"
                )

                if movement is None:
                    continue

                joint_metrics = (
                    movement.get("joints", {})
                    .get(joint_name, {})
                )

                value = joint_metrics.get(
                    metric_name
                )

                if value is not None:
                    values.append(value)

            baseline["joints"][joint_name][
                metric_name
            ] = (
                statistics.mean(values)
                if values
                else None
            )

    # -----------------------------------------
    # Compensation baseline
    # -----------------------------------------

    trunk_values = [
        rep.get("compensation", {}).get(
            "maximum_trunk_displacement_norm"
        )
        for rep in valid_repetitions
        if rep.get("compensation") is not None
    ]

    trunk_values = [
        value
        for value in trunk_values
        if value is not None
    ]

    baseline["compensation"][
        "maximum_trunk_displacement_norm"
    ] = (
        statistics.mean(trunk_values)
        if trunk_values
        else None
    )

    return baseline