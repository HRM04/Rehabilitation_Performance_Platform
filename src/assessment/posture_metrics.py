"""Posture-related movement measurements for Version 1.

These are geometric measurements, not clinical diagnoses
or validated compensation classifications.
"""


def calculate_shoulder_elevation(
    baseline_landmarks,
    current_landmarks,
    min_visibility=0.5,
):
    """Measure upward shoulder movement relative to the hips.

    Returns normalized image-coordinate changes for each side.
    Positive values indicate upward shoulder movement, that is increased
    shoulder-to-hip vertical separation compared with baseline (separation
    is hip y minus shoulder y, and image y grows downward).

    Assumes a consistently oriented, approximately frontal camera.
    """

    if not baseline_landmarks or not current_landmarks:
        return None

    result = {}

    for side in ("left", "right"):
        shoulder = f"{side}_shoulder"
        hip = f"{side}_hip"

        for landmarks in (baseline_landmarks, current_landmarks):
            if any(
                name not in landmarks
                or landmarks[name].get("visibility", 0) < min_visibility
                for name in (shoulder, hip)
            ):
                return None

        baseline_separation = (
            baseline_landmarks[hip]["y"]
            - baseline_landmarks[shoulder]["y"]
        )

        current_separation = (
            current_landmarks[hip]["y"]
            - current_landmarks[shoulder]["y"]
        )

        result[f"{side}_shoulder_elevation_norm"] = (
            current_separation - baseline_separation
        )

    return result


def calculate_trunk_posture(
    baseline_landmarks,
    current_landmarks,
    min_visibility=0.5,
):
    """Measure 2D trunk shift and inclination relative to baseline.

    Assumes an approximately frontal, stationary camera.
    Returns None when required landmarks are unavailable.

    These are image-based posture descriptors, not validated
    compensation classifications.
    """
    import math

    required = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )

    for landmarks in (baseline_landmarks, current_landmarks):
        if not landmarks:
            return None

        if any(
            name not in landmarks
            or landmarks[name].get("visibility", 0) < min_visibility
            for name in required
        ):
            return None

    def geometry(landmarks):
        shoulder_x = (
            landmarks["left_shoulder"]["x"]
            + landmarks["right_shoulder"]["x"]
        ) / 2

        shoulder_y = (
            landmarks["left_shoulder"]["y"]
            + landmarks["right_shoulder"]["y"]
        ) / 2

        hip_x = (
            landmarks["left_hip"]["x"]
            + landmarks["right_hip"]["x"]
        ) / 2

        hip_y = (
            landmarks["left_hip"]["y"]
            + landmarks["right_hip"]["y"]
        ) / 2

        inclination = math.degrees(
            math.atan2(
                shoulder_x - hip_x,
                hip_y - shoulder_y,
            )
        )

        return {
            "trunk_center_x": (shoulder_x + hip_x) / 2,
            "inclination_deg": inclination,
        }

    baseline = geometry(baseline_landmarks)
    current = geometry(current_landmarks)

    return {
        "lateral_shift_norm": (
            current["trunk_center_x"]
            - baseline["trunk_center_x"]
        ),
        "inclination_change_deg": (
            current["inclination_deg"]
            - baseline["inclination_deg"]
        ),
    }


def calculate_shoulder_vertical_displacement(
    baseline_landmarks,
    current_landmarks,
    min_visibility=0.5,
):
    """
    Measure shoulder vertical displacement relative to calibration.

    Uses normalized image coordinates.
    Positive displacement indicates upward movement in the image.

    This is a descriptive image-plane measurement, not a
    clinically validated compensation classification.

    Requires a stationary camera and consistent positioning.
    """

    if not baseline_landmarks or not current_landmarks:
        return None

    result = {}

    for side in ("left", "right"):
        name = f"{side}_shoulder"

        baseline = baseline_landmarks.get(name)
        current = current_landmarks.get(name)

        if not baseline or not current:
            result[f"{side}_vertical_displacement_norm"] = None
            continue

        if (
            baseline.get("visibility", 0) < min_visibility
            or current.get("visibility", 0) < min_visibility
        ):
            result[f"{side}_vertical_displacement_norm"] = None
            continue

        result[f"{side}_vertical_displacement_norm"] = (
            baseline["y"] - current["y"]
        )

    if all(value is None for value in result.values()):
        return None

    return result
