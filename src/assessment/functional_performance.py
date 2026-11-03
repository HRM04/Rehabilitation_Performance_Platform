def assess_functional_performance(
    repetition,
    trajectory_metrics=None,
    minimum_duration_ms=1000,
    minimum_object_movement=20.0,
):
    """
    Assess the basic functional outcome of one repetition.

    A valid attempt must satisfy the minimum duration and
    object-movement requirements.

    If trajectory information is available, successful task
    completion additionally requires the object to finish
    inside the defined target region.

    This is a preliminary engineering assessment and does not
    represent a validated clinical score.
    """

    if repetition is None:
        return {
            "status": "INVALID",
            "valid_attempt": False,
            "completed": False,
            "reason": "Missing repetition data",
        }

    duration_ms = repetition.get("duration_ms")

    total_object_movement = repetition.get(
        "total_object_movement"
    )

    # -------------------------------------------------
    # Basic repetition validity
    # -------------------------------------------------

    if duration_ms is None:
        return {
            "status": "INCOMPLETE",
            "valid_attempt": False,
            "completed": False,
            "reason": "Missing duration",
        }

    if total_object_movement is None:
        return {
            "status": "INCOMPLETE",
            "valid_attempt": False,
            "completed": False,
            "reason": "Missing object movement",
        }

    if duration_ms < minimum_duration_ms:
        return {
            "status": "INCOMPLETE",
            "valid_attempt": False,
            "completed": False,
            "reason": "Interaction too short",
        }

    if total_object_movement < minimum_object_movement:
        return {
            "status": "INCOMPLETE",
            "valid_attempt": False,
            "completed": False,
            "reason": "Insufficient object movement",
        }

    # -------------------------------------------------
    # Valid attempt, but no target information
    # -------------------------------------------------

    if trajectory_metrics is None:
        return {
            "status": "VALID_ATTEMPT",
            "valid_attempt": True,
            "completed": False,
            "reason": "Target completion not assessed",
            "duration_ms": duration_ms,
            "total_object_movement": total_object_movement,
        }

    # -------------------------------------------------
    # Target-based task completion
    # -------------------------------------------------

    target_reached = trajectory_metrics.get(
        "target_reached"
    )

    if target_reached is True and repetition.get(
        "ended_with_recording"
    ) is True:
        return {
            "status": "INCOMPLETE",
            "valid_attempt": True,
            "completed": False,
            "reason": (
                "Target reached, but recording ended "
                "without a confirmed repetition endpoint"
            ),
            "duration_ms": duration_ms,
            "total_object_movement": total_object_movement,
            "target_reached": True,
            "final_target_error_px": trajectory_metrics.get(
                "final_target_error_px"
            ),
        }

    if target_reached is True:
        return {
            "status": "COMPLETED",
            "valid_attempt": True,
            "completed": True,
            "reason": None,
            "duration_ms": duration_ms,
            "total_object_movement": total_object_movement,
            "target_reached": True,
            "final_target_error_px":
                trajectory_metrics.get(
                    "final_target_error_px"
                ),
        }

    return {
        "status": "TARGET_NOT_REACHED",
        "valid_attempt": True,
        "completed": False,
        "reason": "Object did not finish inside target region",
        "duration_ms": duration_ms,
        "total_object_movement": total_object_movement,
        "target_reached": False,
        "final_target_error_px":
            trajectory_metrics.get(
                "final_target_error_px"
            ),
    }