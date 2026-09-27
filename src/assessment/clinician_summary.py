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