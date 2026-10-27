import statistics

from src.assessment.posture_calibration import (
    REQUIRED_LANDMARKS,
    OPTIONAL_LANDMARKS,
    calibration_subset,
    usable_frames,
)


def assess_posture_stability(
    calibration_frames,
    min_valid_frames=10,
    min_visibility=0.5,
    maximum_mad_norm=0.015,
):
    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be at least 1")
    if maximum_mad_norm < 0:
        raise ValueError("maximum_mad_norm must be non-negative")

    frames, names, mode = calibration_subset(
        calibration_frames, min_valid_frames, min_visibility
    )
    if frames is None:
        return {
            "status": "INSUFFICIENT_DATA",
            "valid_frames": len(
                usable_frames(calibration_frames, min_visibility)
            ),
            "stable": None,
        }

    def compute_mad(observations, landmarks):
        variability = {}
        for name in landmarks:
            axes = {}
            for axis in ("x", "y"):
                values = [float(f[name][axis]) for f in observations]
                median = statistics.median(values)
                axes[axis] = statistics.median(
                    abs(v - median) for v in values
                )
            variability[name] = axes
        maximum = max(
            v for axes in variability.values() for v in axes.values()
        )
        return variability, maximum

    variability, maximum = compute_mad(frames, names)

    # When hips are visible but unstable, retain a stable
    # shoulder-only reference instead of failing the task.
    if mode == "FULL" and maximum > maximum_mad_norm:
        shoulder_frames = usable_frames(
            calibration_frames, min_visibility
        )
        shoulder_variability, shoulder_max = compute_mad(
            shoulder_frames, REQUIRED_LANDMARKS
        )
        if shoulder_max <= maximum_mad_norm:
            frames = shoulder_frames
            variability = shoulder_variability
            maximum = shoulder_max
            mode = "PARTIAL"

    stable = maximum <= maximum_mad_norm
    return {
        "status": "STABLE" if stable else "UNSTABLE",
        "stable": stable,
        "valid_frames": len(frames),
        "maximum_mad_norm": maximum,
        "threshold_mad_norm": maximum_mad_norm,
        "landmark_variability": variability,
        "measurement_type": "image_landmark_position_variability",
        "calibration_mode": mode,
    }
