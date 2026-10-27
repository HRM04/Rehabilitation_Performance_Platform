def create_repetition_assessment(
    repetition,
    movement_metrics=None,
    functional_performance=None,
    grasp_assessment=None,
    trajectory_metrics=None,
):
    """
    Combine all available assessment information for one
    rehabilitation repetition.

    Missing modalities remain None rather than being
    interpreted as zero or normal performance.
    """

    if repetition is None:
        return None

    assessment = {
        "repetition_number": repetition.get(
            "repetition_number"
        ),

        "timing": {
            "start_frame": repetition.get("start_frame"),
            "end_frame": repetition.get("end_frame"),
            "start_timestamp_ms": repetition.get(
                "start_timestamp_ms"
            ),
            "end_timestamp_ms": repetition.get(
                "end_timestamp_ms"
            ),
            "duration_ms": repetition.get("duration_ms"),

            "ended_with_recording": repetition.get(
                "ended_with_recording", False
            ),
        },

        "object": {
            "total_movement": repetition.get(
                "total_object_movement"
            ),
        },

        "functional_performance": functional_performance,

        "movement_quality": movement_metrics,

        "grasp_assessment": grasp_assessment,

        "trajectory": trajectory_metrics,

        "compensation": None,
    }

    # Trunk movement is currently our preliminary
    # compensation-related measurement.
    if movement_metrics is not None:

        trunk_metrics = movement_metrics.get("trunk")

        if trunk_metrics is not None:
            assessment["compensation"] = {
                "maximum_trunk_displacement_norm":
                    trunk_metrics.get(
                        "maximum_displacement_norm"
                    ),

                "valid_trunk_frames":
                    trunk_metrics.get(
                        "valid_frames"
                    ),

                "mean_trunk_displacement_norm":
                    trunk_metrics.get(
                        "mean_displacement_norm"
                    ),

                "trunk_displacement_std_norm":
                    trunk_metrics.get(
                        "displacement_std_norm"
                    ),

                "peak_trunk_displacement_fraction":
                    trunk_metrics.get(
                        "peak_position_fraction"
                    ),

                "measurement_type": "trunk_movement",
                "compensation_confirmed": False,

                "shoulder_elevation": (
                    movement_metrics.get("shoulder_elevation")
                ),

                "trunk_posture": (
                    movement_metrics.get("trunk_posture")
                ),
            }

    return assessment