import math


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b formed by three 2D points.

    Example:
        Shoulder -> Elbow -> Wrist
        The resulting angle is the elbow angle.
    """

    ax, ay = point_a["x"], point_a["y"]
    bx, by = point_b["x"], point_b["y"]
    cx, cy = point_c["x"], point_c["y"]

    vector_1 = (ax - bx, ay - by)
    vector_2 = (cx - bx, cy - by)

    dot_product = (
        vector_1[0] * vector_2[0]
        + vector_1[1] * vector_2[1]
    )

    magnitude_1 = math.hypot(*vector_1)
    magnitude_2 = math.hypot(*vector_2)

    if magnitude_1 == 0 or magnitude_2 == 0:
        return None

    cosine_angle = dot_product / (magnitude_1 * magnitude_2)

    # Prevent floating-point rounding from causing acos() errors.
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))



def calculate_upper_body_angles(landmarks, min_visibility=0.5):
    """
    Calculate preliminary 2D elbow and shoulder angles.

    Return None for an angle if any required landmark is missing
    or has insufficient visibility.
    """

    if landmarks is None:
        return {
            "left_elbow": None,
            "right_elbow": None,
            "left_shoulder": None,
            "right_shoulder": None,
        }

    def get_angle(point_a, point_b, point_c):
        names = (point_a, point_b, point_c)

        if any(
            name not in landmarks
            or landmarks[name]["visibility"] < min_visibility
            for name in names
        ):
            return None

        return calculate_angle(
            landmarks[point_a],
            landmarks[point_b],
            landmarks[point_c],
        )

    return {
        "left_elbow": get_angle(
            "left_shoulder", "left_elbow", "left_wrist"
        ),
        "right_elbow": get_angle(
            "right_shoulder", "right_elbow", "right_wrist"
        ),
        "left_shoulder": get_angle(
            "left_hip", "left_shoulder", "left_elbow"
        ),
        "right_shoulder": get_angle(
            "right_hip", "right_shoulder", "right_elbow"
        ),
    }