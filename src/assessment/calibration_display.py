"""Normalize calibration display fields across assessment versions."""


def calibration_view(calibration):
    if not isinstance(calibration, dict):
        calibration = {}

    reference = calibration.get("reference_posture")
    if not isinstance(reference, dict):
        reference = {}

    stability = calibration.get("stability")
    if not isinstance(stability, dict):
        stability = {}

    status = calibration.get("status", "NOT_RECORDED")

    available = bool(
        calibration.get("reference_available", bool(reference))
    )

    if status != "READY":
        available = False

    reference_type = (
        calibration.get("reference_type")
        or reference.get("reference_type")
    )

    return {
        "status": status,
        "reference_available": available,
        "reference_type": reference_type or "Not recorded",
        "stability_status": stability.get("status", "Not recorded"),
        "baseline": calibration.get("assessment_baseline") or (
            "calibrated_reference"
            if available else "first_valid_task_frame"
        ),
        "reason": calibration.get("reason"),
    }
