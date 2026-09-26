def calculate_range_of_motion(angle_values):
    """
    Calculate the observed angular range of motion.

    Args:
        angle_values: List of joint angles in degrees.
                      Missing values may be represented by None.

    Returns:
        Dictionary containing minimum angle, maximum angle,
        and observed angular range in degrees.
    """

    valid_angles = [
        angle
        for angle in angle_values
        if angle is not None
    ]

    if len(valid_angles) < 2:
        return {
            "min_angle": None,
            "max_angle": None,
            "range_of_motion": None,
        }

    min_angle = min(valid_angles)
    max_angle = max(valid_angles)

    return {
        "min_angle": min_angle,
        "max_angle": max_angle,
        "range_of_motion": max_angle - min_angle,
    }




def calculate_angular_velocity(
    previous_angle,
    current_angle,
    previous_timestamp_ms,
    current_timestamp_ms,
):
    """
    Calculate angular velocity between two observations.

    Returns signed angular velocity in degrees per second.
    A positive value means the measured angle increased.
    """

    if previous_angle is None or current_angle is None:
        return None

    if previous_timestamp_ms is None or current_timestamp_ms is None:
        return None

    elapsed_seconds = (
        current_timestamp_ms - previous_timestamp_ms
    ) / 1000.0

    if elapsed_seconds <= 0:
        return None

    return (
        current_angle - previous_angle
    ) / elapsed_seconds


def calculate_angular_acceleration(
    previous_velocity,
    current_velocity,
    previous_timestamp_ms,
    current_timestamp_ms,
):
    """
    Calculate angular acceleration in degrees per second squared.

    Returns None when the measurements or time interval are invalid.
    """

    if previous_velocity is None or current_velocity is None:
        return None

    if previous_timestamp_ms is None or current_timestamp_ms is None:
        return None

    elapsed_seconds = (
        current_timestamp_ms - previous_timestamp_ms
    ) / 1000.0

    if elapsed_seconds <= 0:
        return None

    return (
        current_velocity - previous_velocity
    ) / elapsed_seconds




def count_velocity_reversals(velocities, noise_threshold=5.0):
    """
    Count changes in angular movement direction.

    Velocities are in degrees/second.
    Values within +/- noise_threshold are ignored.
    Missing values break continuity.
    """
    reversal_count = 0
    previous_direction = None

    for velocity in velocities:
        if velocity is None:
            previous_direction = None
            continue

        if abs(velocity) <= noise_threshold:
            continue

        current_direction = 1 if velocity > 0 else -1

        if (
            previous_direction is not None
            and current_direction != previous_direction
        ):
            reversal_count += 1

        previous_direction = current_direction

    return reversal_count



def calculate_trunk_center(landmarks, min_visibility=0.5):
    """
    Estimate the trunk centre using the shoulder and hip landmarks.

    Returns normalized image coordinates or None when the
    required landmarks are unavailable.
    """

    if landmarks is None:
        return None

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    if any(
        name not in landmarks
        or landmarks[name]["visibility"] < min_visibility
        for name in required
    ):
        return None

    return {
        "x": sum(landmarks[name]["x"] for name in required) / 4,
        "y": sum(landmarks[name]["y"] for name in required) / 4,
    }



import math


def calculate_trunk_displacement(
    baseline_center,
    current_center,
):
    """
    Calculate 2D trunk displacement relative to a baseline.

    Returns the displacement in normalized image coordinates,
    or None if either position is unavailable.
    """

    if baseline_center is None or current_center is None:
        return None

    dx = current_center["x"] - baseline_center["x"]
    dy = current_center["y"] - baseline_center["y"]

    return math.hypot(dx, dy)