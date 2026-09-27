def generate_feedback(
    baseline_comparison,
    rom_drop_threshold=-15.0,
    speed_drop_threshold=-20.0,
    duration_increase_threshold=20.0,
    trunk_increase_threshold=25.0,
    reversal_increase_threshold=25.0,
):
    """
    Generate preliminary rule-based rehabilitation feedback
    from patient-specific baseline comparisons.

    Thresholds are engineering defaults for prototype use only.
    They are not validated clinical thresholds.
    """

    if baseline_comparison is None:
        return []

    feedback = []

    # -----------------------------------------
    # Duration
    # -----------------------------------------

    duration_change = baseline_comparison.get(
        "duration_change_percent"
    )

    if (
        duration_change is not None
        and duration_change >= duration_increase_threshold
    ):
        feedback.append({
            "category": "task_duration",
            "message": (
                "Task execution was slower than the "
                "patient-specific baseline."
            ),
            "change_percent": duration_change,
        })

    # -----------------------------------------
    # Joint movement
    # -----------------------------------------

    joints = baseline_comparison.get("joints", {})

    for joint_name, metrics in joints.items():

        rom_change = metrics.get("rom_deg")

        if (
            rom_change is not None
            and rom_change <= rom_drop_threshold
        ):
            feedback.append({
                "category": "range_of_motion",
                "joint": joint_name,
                "message": (
                    f"{joint_name} range of motion was "
                    "reduced relative to baseline."
                ),
                "change_percent": rom_change,
            })

        speed_change = metrics.get(
            "mean_angular_speed_deg_s"
        )

        if (
            speed_change is not None
            and speed_change <= speed_drop_threshold
        ):
            feedback.append({
                "category": "movement_speed",
                "joint": joint_name,
                "message": (
                    f"{joint_name} movement speed was "
                    "reduced relative to baseline."
                ),
                "change_percent": speed_change,
            })

        reversal_change = metrics.get(
            "velocity_reversals"
        )

        if (
            reversal_change is not None
            and reversal_change >= reversal_increase_threshold
        ):
            feedback.append({
                "category": "movement_control",
                "joint": joint_name,
                "message": (
                    f"{joint_name} showed more movement "
                    "direction changes than baseline."
                ),
                "change_percent": reversal_change,
            })

    # -----------------------------------------
    # Compensation-related movement
    # -----------------------------------------

    trunk_change = (
        baseline_comparison
        .get("compensation", {})
        .get("trunk_displacement_change_percent")
    )

    if (
        trunk_change is not None
        and trunk_change >= trunk_increase_threshold
    ):
        feedback.append({
            "category": "trunk_movement",
            "message": (
                "Trunk displacement increased relative "
                "to the patient-specific baseline."
            ),
            "change_percent": trunk_change,
        })

    # -----------------------------------------
    # No flagged change
    # -----------------------------------------

    if not feedback:
        feedback.append({
            "category": "no_flagged_change",
            "message": (
                "No large deviation from the current "
                "patient-specific baseline was detected "
                "using the configured prototype thresholds."
            ),
        })

    return feedback