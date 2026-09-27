def assess_functional_performance(
    repetition,
    minimum_duration_ms=1000,
    minimum_object_movement=20.0,
):
    """
    Assess the basic functional outcome of one repetition.

    This is a preliminary engineering assessment and does not
    represent a validated clinical score.
    """

    if repetition is None:
        return {
            "status": "INVALID",
            "completed": False,
            "reason": "Missing repetition data",
        }

    duration_ms = repetition.get("duration_ms")
    total_object_movement = repetition.get(
        "total_object_movement"
    )

    if duration_ms is None:
        return {
            "status": "INCOMPLETE",
            "completed": False,
            "reason": "Missing duration",
        }

    if total_object_movement is None:
        return {
            "status": "INCOMPLETE",
            "completed": False,
            "reason": "Missing object movement",
        }

    if duration_ms < minimum_duration_ms:
        return {
            "status": "INCOMPLETE",
            "completed": False,
            "reason": "Interaction too short",
        }

    if total_object_movement < minimum_object_movement:
        return {
            "status": "INCOMPLETE",
            "completed": False,
            "reason": "Insufficient object movement",
        }

    return {
        "status": "COMPLETED_ATTEMPT",
        "completed": True,
        "reason": None,
        "duration_ms": duration_ms,
        "total_object_movement": total_object_movement,
    }