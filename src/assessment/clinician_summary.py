def create_clinician_summary(
    session_id,
    task_id,
    repetition_assessments,
    consistency_results=None,
    deterioration_results=None,
    feedback=None,
):
    """
    Create a clinician-facing summary structure for one
    rehabilitation session.

    This structure organizes available prototype outputs.
    It does not provide a clinical diagnosis.
    """

    valid_repetitions = [
        rep
        for rep in repetition_assessments
        if rep is not None
    ]

    completed_repetitions = [
        rep
        for rep in valid_repetitions
        if (
            rep.get("functional_performance", {})
            .get("completed") is True
        )
    ]

    summary = {
        "session_id": session_id,
        "task_id": task_id,

        "repetition_summary": {
            "total_detected": len(valid_repetitions),
            "completed_attempts": len(
                completed_repetitions
            ),
        },

        "trajectory_summary": {
            "available": False,
            "results": [],
        },

        "grasp_summary": {
            "available": False,
            "results": [],
        },

        "movement_summary": {
            "consistency": consistency_results,
            "performance_change":
                deterioration_results,
        },

        "compensation_summary": {
            "maximum_trunk_displacements": [],
        },

        "feedback": feedback or [],
    }


    # -----------------------------------------
    # Trajectory results
    # -----------------------------------------

    for rep in valid_repetitions:

        trajectory = rep.get("trajectory")

        if trajectory is None:
            continue

        summary["trajectory_summary"][
            "available"
        ] = True

        summary["trajectory_summary"][
            "results"
        ].append({
            "repetition_number":
                rep.get("repetition_number"),

            "path_length_px":
                trajectory.get(
                    "path_length_px"
                ),

            "direct_displacement_px":
                trajectory.get(
                    "direct_displacement_px"
                ),

            "movement_efficiency":
                trajectory.get(
                    "movement_efficiency"
                ),

            "mean_trajectory_deviation_px":
                trajectory.get(
                    "mean_trajectory_deviation_px"
                ),

            "final_target_error_px":
                trajectory.get(
                    "final_target_error_px"
                ),

            "target_reached":
                trajectory.get(
                    "target_reached"
                ),
        })

    # -----------------------------------------
    # Grasp results
    # -----------------------------------------

    for rep in valid_repetitions:

        grasp = rep.get("grasp_assessment")

        if grasp is not None:
            summary["grasp_summary"][
                "available"
            ] = True

            summary["grasp_summary"][
                "results"
            ].append(grasp)

    # -----------------------------------------
    # Trunk movement
    # -----------------------------------------

    for rep in valid_repetitions:

        compensation = rep.get(
            "compensation"
        )

        if compensation is None:
            continue

        displacement = compensation.get(
            "maximum_trunk_displacement_norm"
        )

        if displacement is not None:
            summary["compensation_summary"][
                "maximum_trunk_displacements"
            ].append(displacement)

    return summary