from src.assessment.posture_calibration import (
    calibrate_reference_posture,
)

from src.assessment.posture_stability import (
    assess_posture_stability,
)


def prepare_posture_calibration(
    calibration_frames,
    min_valid_frames=10,
    min_visibility=0.5,
    maximum_mad_norm=0.015,
):
    """
    Prepare a posture reference from a dedicated
    pre-task calibration recording.

    The caller must supply calibration frames recorded
    separately from rehabilitation task repetitions.
    """

    if calibration_frames is None:
        return {
            "status": "UNAVAILABLE",
            "reference_posture": None,
            "reason": "No calibration recording supplied",
        }

    stability = assess_posture_stability(
        calibration_frames,
        min_valid_frames=min_valid_frames,
        min_visibility=min_visibility,
        maximum_mad_norm=maximum_mad_norm,
    )

    if stability["status"] == "INSUFFICIENT_DATA":
        return {
            "status": "INSUFFICIENT_DATA",
            "reference_posture": None,
            "reason": "Insufficient valid posture frames",
            "total_frames": len(calibration_frames),
            "valid_frames": stability["valid_frames"],
            "required_valid_frames": min_valid_frames,
        }

    if stability["status"] == "UNSTABLE":
        return {
            "status": "UNSTABLE",
            "reference_posture": None,
            "reason": "Excessive landmark movement during calibration",
            "total_frames": len(calibration_frames),
            "valid_frames": stability["valid_frames"],
            "stability": stability,
        }

    reference_frames = calibration_frames
    if stability.get("calibration_mode") == "PARTIAL":
        # Suppress optional hips in the reference to match
        # the validated shoulder-only stability result.
        reference_frames = [
            {
                **frame,
                "landmarks": {
                    name: point
                    for name, point in (frame.get("landmarks") or {}).items()
                    if name in ("left_shoulder", "right_shoulder")
                },
            }
            for frame in calibration_frames
        ]

    reference = calibrate_reference_posture(
        reference_frames,
        min_valid_frames=min_valid_frames,
        min_visibility=min_visibility,
    )

    if reference is None:
        return {
            "status": "INSUFFICIENT_DATA",
            "reference_posture": None,
            "reason": "Insufficient valid posture frames",
            "total_frames": len(calibration_frames),
            "required_valid_frames": min_valid_frames,
        }

    return {
        "status": "READY",
        "calibration_mode": reference["calibration_mode"],
        "trunk_reference_available":
            reference["trunk_reference_available"],
        "shoulder_elevation_reference_available":
            reference["shoulder_elevation_reference_available"],
        "unavailable_measurements":
            reference["unavailable_measurements"],
        "reference_posture": reference,
        "total_frames": len(calibration_frames),
        "valid_frames": reference["valid_frames"],
        "reference_type": reference["reference_type"],
        "method": reference["method"],
        "stability": stability,
    }
