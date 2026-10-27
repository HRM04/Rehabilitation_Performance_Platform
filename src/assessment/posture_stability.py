import statistics

from src.assessment.posture_calibration import (
    POSTURE_LANDMARKS,
)


def assess_posture_stability(
    calibration_frames,
    min_valid_frames=10,
    min_visibility=0.5,
    maximum_mad_norm=0.015,
):
    """
    Assess landmark-position stability during posture calibration.

    Uses median absolute deviation (MAD) of normalized
    image coordinates.

    Thresholds are provisional engineering parameters,
    not clinically validated posture criteria.
    """

    if maximum_mad_norm < 0:
        raise ValueError("maximum_mad_norm must be non-negative")

    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be at least 1")

    valid_frames = []

    for frame in calibration_frames or []:
        landmarks = frame.get("landmarks") or {}

        if any(
            name not in landmarks
            or landmarks[name].get("visibility", 0) < min_visibility
            or not isinstance(landmarks[name].get("x"), (int, float))
            or not isinstance(landmarks[name].get("y"), (int, float))
            for name in POSTURE_LANDMARKS
        ):
            continue

        valid_frames.append(landmarks)

    if len(valid_frames) < min_valid_frames:
        return {
            "status": "INSUFFICIENT_DATA",
            "valid_frames": len(valid_frames),
            "stable": None,
        }

    landmark_variability = {}

    for name in POSTURE_LANDMARKS:
        coordinate_mads = {}

        for axis in ("x", "y"):
            values = [
                frame[name][axis]
                for frame in valid_frames
            ]

            median = statistics.median(values)

            coordinate_mads[axis] = statistics.median(
                abs(value - median)
                for value in values
            )

        landmark_variability[name] = coordinate_mads

    maximum_observed_mad = max(
        value
        for axes in landmark_variability.values()
        for value in axes.values()
    )

    stable = maximum_observed_mad <= maximum_mad_norm

    return {
        "status": "STABLE" if stable else "UNSTABLE",
        "stable": stable,
        "valid_frames": len(valid_frames),
        "maximum_mad_norm": maximum_observed_mad,
        "threshold_mad_norm": maximum_mad_norm,
        "landmark_variability": landmark_variability,
        "measurement_type": "image_landmark_position_variability",
    }
