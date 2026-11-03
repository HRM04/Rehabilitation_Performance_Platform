from src.grasp.grasp_features import (
    calculate_thumb_index_distance,
    calculate_finger_angles,
    calculate_thumb_opposition
)



PINCH_REFERENCE = 0.257
OPEN_REFERENCE = 0.875


def classify_pinch(thumb_index_distance):

    pinch_score = (
        (OPEN_REFERENCE - thumb_index_distance)
        / (OPEN_REFERENCE - PINCH_REFERENCE)
    )

    # Keep score between 0 and 1
    pinch_score = max(0.0, min(1.0, pinch_score))

    pinch_percentage = pinch_score * 100

    if pinch_percentage >= 50:
        predicted_grasp = "PINCH"
        confidence = pinch_percentage

    else:
        predicted_grasp = "NOT PINCH"
        confidence = 100 - pinch_percentage

    return predicted_grasp, confidence



def detect_cylindrical_grasp(hand_data):
    """
    Detect a cylindrical-like hand configuration.

    This is an initial rule-based heuristic.
    Thresholds will be calibrated experimentally.
    """

    finger_angles = calculate_finger_angles(hand_data)
    thumb_distances = calculate_thumb_opposition(hand_data)

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 150
        for angle in valid_angles
    )

    strongly_flexed_fingers = sum(
        angle < 120
        for angle in valid_angles
    )

    thumb_opposed = (
        thumb_distances["thumb_index"] < 1.5
        or thumb_distances["thumb_middle"] < 1.5
    )

    finger_score = flexed_fingers / 4.0

    strong_flexion_score = strongly_flexed_fingers / 4.0

    thumb_score = 1.0 if thumb_opposed else 0.0

    cylindrical_score = (
        0.50 * finger_score
        + 0.25 * strong_flexion_score
        + 0.25 * thumb_score
    )

    confidence = cylindrical_score * 100

    is_cylindrical = (
        flexed_fingers >= 3
        and thumb_opposed
        and confidence >= 60
    )

    return is_cylindrical, confidence




def detect_spherical_grasp(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance,
    finger_spread
):
    """
    Detect a spherical-like grasp configuration.

    Initial heuristic only.
    Thresholds will be calibrated experimentally.
    """

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 155
        for angle in valid_angles
    )

    strongly_flexed_fingers = sum(
        angle < 125
        for angle in valid_angles
    )

    thumb_opposed = (
        thumb_index_distance < 1.7
        or thumb_middle_distance < 1.7
    )

    average_spread = sum(
        finger_spread.values()
    ) / len(finger_spread)

    fingers_spread = average_spread > 0.20

    finger_score = flexed_fingers / 4.0
    strong_flexion_score = strongly_flexed_fingers / 4.0
    thumb_score = 1.0 if thumb_opposed else 0.0
    spread_score = 1.0 if fingers_spread else 0.0

    spherical_score = (
        0.40 * finger_score
        + 0.20 * strong_flexion_score
        + 0.20 * thumb_score
        + 0.20 * spread_score
    )

    confidence = spherical_score * 100

    is_spherical = (
        flexed_fingers >= 3
        and thumb_opposed
        and fingers_spread
        and confidence >= 60
    )

    return is_spherical, confidence





def detect_large_aperture_palmar_grasp(
    finger_angles,
    finger_spread,
    hand_aperture,
    thumb_index_distance,
    thumb_middle_distance
):
    """
    Detect a large-aperture palmar grasp.

    Initial rule-based heuristic.
    Thresholds will be calibrated experimentally.
    """

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 160
        for angle in valid_angles
    )

    average_spread = (
        sum(finger_spread.values())
        / len(finger_spread)
    )

    fingers_spread = average_spread > 0.18

    large_aperture = hand_aperture > 1.0

    thumb_engaged = (
        thumb_index_distance < 2.0
        or thumb_middle_distance < 2.0
    )

    finger_score = flexed_fingers / 4.0
    spread_score = 1.0 if fingers_spread else 0.0
    aperture_score = 1.0 if large_aperture else 0.0
    thumb_score = 1.0 if thumb_engaged else 0.0

    palmar_score = (
        0.35 * finger_score
        + 0.25 * spread_score
        + 0.25 * aperture_score
        + 0.15 * thumb_score
    )

    confidence = palmar_score * 100

    is_large_palmar = (
        flexed_fingers >= 3
        and fingers_spread
        and large_aperture
        and confidence >= 60
    )

    return is_large_palmar, confidence





def detect_whole_hand_power_grip(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance,
    hand_aperture
):
    """
    Detect a whole-hand power grip.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 145
        for angle in valid_angles
    )

    strongly_flexed_fingers = sum(
        angle < 110
        for angle in valid_angles
    )

    thumb_engaged = (
        thumb_index_distance < 1.6
        or thumb_middle_distance < 1.6
    )

    closed_hand = hand_aperture < 1.5

    finger_score = flexed_fingers / 4.0
    strong_flexion_score = strongly_flexed_fingers / 4.0
    thumb_score = 1.0 if thumb_engaged else 0.0
    closure_score = 1.0 if closed_hand else 0.0

    power_score = (
        0.40 * finger_score
        + 0.30 * strong_flexion_score
        + 0.15 * thumb_score
        + 0.15 * closure_score
    )

    confidence = power_score * 100

    is_power_grip = (
        flexed_fingers >= 3
        and strongly_flexed_fingers >= 2
        and thumb_engaged
        and closed_hand
        and confidence >= 60
    )

    return is_power_grip, confidence




def detect_handle_grasp(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance,
    hand_aperture,
    finger_spread
):
    """
    Detect a handle-like grasp configuration.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 150
        for angle in valid_angles
    )

    strongly_flexed_fingers = sum(
        angle < 120
        for angle in valid_angles
    )

    thumb_engaged = (
        thumb_index_distance < 1.7
        or thumb_middle_distance < 1.7
    )

    average_spread = (
        sum(finger_spread.values())
        / len(finger_spread)
    )

    fingers_grouped = average_spread < 0.35

    moderate_aperture = (
        0.35 < hand_aperture < 1.4
    )

    finger_score = flexed_fingers / 4.0
    strong_flexion_score = strongly_flexed_fingers / 4.0
    thumb_score = 1.0 if thumb_engaged else 0.0
    grouping_score = 1.0 if fingers_grouped else 0.0
    aperture_score = 1.0 if moderate_aperture else 0.0

    handle_score = (
        0.35 * finger_score
        + 0.20 * strong_flexion_score
        + 0.20 * thumb_score
        + 0.15 * grouping_score
        + 0.10 * aperture_score
    )

    confidence = handle_score * 100

    is_handle_grasp = (
        flexed_fingers >= 3
        and thumb_engaged
        and fingers_grouped
        and moderate_aperture
        and confidence >= 60
    )

    return is_handle_grasp, confidence





def detect_hook_grasp(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance,
    hand_aperture
):
    """
    Detect a hook-like grasp configuration.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    flexed_fingers = sum(
        angle < 140
        for angle in valid_angles
    )

    strongly_flexed_fingers = sum(
        angle < 110
        for angle in valid_angles
    )

    thumb_not_opposed = (
        thumb_index_distance > 0.8
        and thumb_middle_distance > 0.8
    )

    closed_hand = hand_aperture < 1.8

    finger_score = flexed_fingers / 4.0
    strong_flexion_score = strongly_flexed_fingers / 4.0
    thumb_score = 1.0 if thumb_not_opposed else 0.0
    closure_score = 1.0 if closed_hand else 0.0

    hook_score = (
        0.40 * finger_score
        + 0.35 * strong_flexion_score
        + 0.15 * thumb_score
        + 0.10 * closure_score
    )

    confidence = hook_score * 100

    is_hook_grasp = (
        flexed_fingers >= 3
        and strongly_flexed_fingers >= 2
        and thumb_not_opposed
        and confidence >= 60
    )

    return is_hook_grasp, confidence



def detect_pincer_grasp(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance
):
    """
    Detect a thumb-index pincer (pad-to-pad) configuration.

    Uses thumb-index proximity, index flexion, and separation
    from tripod-like configurations.

    Returns (detected, heuristic_score).
    """
    required = ("index", "middle", "ring", "little")

    if any(finger_angles.get(name) is None for name in required):
        return False, 0.0

    index_angle = finger_angles["index"]
    other_angles = [
        finger_angles["middle"],
        finger_angles["ring"],
        finger_angles["little"],
    ]

    thumb_index_close = thumb_index_distance < 0.55
    index_primary = (
        thumb_index_distance + 0.10 < thumb_middle_distance
    )

    # Separate pad-to-pad pincer from the more extended
    # index configuration used for tip pinch.
    index_pad_configuration = 70 <= index_angle <= 120

    # Avoid classifying a three-digit grasp as a two-digit pinch.
    middle_not_participating = thumb_middle_distance >= 0.65

    other_fingers_not_dominant = (
        sum(angle < 110 for angle in other_angles) <= 2
    )

    distance_score = max(
        0.0,
        min(1.0, (0.875 - thumb_index_distance) / 0.618)
    )

    score = 100.0 * (
        0.40 * distance_score
        + 0.20 * float(index_primary)
        + 0.20 * float(index_pad_configuration)
        + 0.10 * float(middle_not_participating)
        + 0.10 * float(other_fingers_not_dominant)
    )

    detected = (
        thumb_index_close
        and index_primary
        and index_pad_configuration
        and middle_not_participating
        and other_fingers_not_dominant
        and score >= 60
    )

    return detected, score


def detect_tip_pinch(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance
):
    """
    Detect a thumb-index tip-pinch configuration.

    Uses fingertip proximity, relatively extended index
    configuration, and exclusion of tripod-like patterns.

    Returns (detected, heuristic_score).
    """
    required = ("index", "middle", "ring", "little")

    if any(finger_angles.get(name) is None for name in required):
        return False, 0.0

    index_angle = finger_angles["index"]
    other_angles = [
        finger_angles["middle"],
        finger_angles["ring"],
        finger_angles["little"],
    ]

    fingertips_close = thumb_index_distance < 0.40
    index_primary = (
        thumb_index_distance + 0.10 < thumb_middle_distance
    )

    index_tip_configuration = index_angle > 120
    middle_not_participating = thumb_middle_distance >= 0.65

    other_fingers_not_dominant = (
        sum(angle < 110 for angle in other_angles) <= 2
    )

    distance_score = max(
        0.0,
        min(1.0, (0.875 - thumb_index_distance) / 0.618)
    )

    score = 100.0 * (
        0.40 * distance_score
        + 0.20 * float(index_primary)
        + 0.20 * float(index_tip_configuration)
        + 0.10 * float(middle_not_participating)
        + 0.10 * float(other_fingers_not_dominant)
    )

    detected = (
        fingertips_close
        and index_primary
        and index_tip_configuration
        and middle_not_participating
        and other_fingers_not_dominant
        and score >= 60
    )

    return detected, score


def detect_tripod_grasp(
    finger_angles,
    thumb_index_distance,
    thumb_middle_distance
):
    """
    Detect a three-jaw chuck / tripod grasp.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    index_angle = finger_angles.get("index")
    middle_angle = finger_angles.get("middle")
    ring_angle = finger_angles.get("ring")
    little_angle = finger_angles.get("little")

    if index_angle is None or middle_angle is None:
        return False, 0.0

    thumb_index_close = (
        thumb_index_distance < 0.65
    )

    thumb_middle_close = (
        thumb_middle_distance < 0.65
    )

    index_engaged = (
        index_angle < 165
    )

    middle_engaged = (
        middle_angle < 165
    )

    other_angles = [
        ring_angle,
        little_angle
    ]

    valid_other_angles = [
        angle
        for angle in other_angles
        if angle is not None
    ]

    strongly_flexed_other_fingers = sum(
        angle < 110
        for angle in valid_other_angles
    )

    other_fingers_not_dominant = (
        strongly_flexed_other_fingers <= 1
    )

    index_distance_score = max(
        0.0,
        min(
            1.0,
            (0.875 - thumb_index_distance)
            / (0.875 - 0.257)
        )
    )

    middle_distance_score = max(
        0.0,
        min(
            1.0,
            (0.875 - thumb_middle_distance)
            / (0.875 - 0.257)
        )
    )

    index_score = 1.0 if index_engaged else 0.0
    middle_score = 1.0 if middle_engaged else 0.0
    isolation_score = (
        1.0 if other_fingers_not_dominant else 0.0
    )

    tripod_score = (
        0.30 * index_distance_score
        + 0.30 * middle_distance_score
        + 0.15 * index_score
        + 0.15 * middle_score
        + 0.10 * isolation_score
    )

    confidence = tripod_score * 100

    is_tripod = (
        thumb_index_close
        and thumb_middle_close
        and index_engaged
        and middle_engaged
        and other_fingers_not_dominant
        and confidence >= 60
    )

    return is_tripod, confidence





def detect_lateral_pinch(
    finger_angles,
    thumb_index_distance,
    thumb_index_side_distance
):
    """
    Detect a lateral / key pinch.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    index_angle = finger_angles.get("index")

    if index_angle is None:
        return False, 0.0

    # Thumb should be relatively close to the side of the index finger
    thumb_near_index_side = (
        thumb_index_side_distance < 0.75
    )

    # But thumb and index fingertips should not necessarily meet
    fingertips_not_too_close = (
        thumb_index_distance > 0.30
    )

    # Index is usually fairly extended or only moderately flexed
    index_position_ok = (
        index_angle > 115
    )

    side_distance_score = max(
        0.0,
        min(
            1.0,
            (1.2 - thumb_index_side_distance)
            / (1.2 - 0.30)
        )
    )

    side_score = 1.0 if thumb_near_index_side else 0.0
    fingertip_score = 1.0 if fingertips_not_too_close else 0.0
    index_score = 1.0 if index_position_ok else 0.0

    lateral_score = (
        0.45 * side_distance_score
        + 0.25 * side_score
        + 0.15 * fingertip_score
        + 0.15 * index_score
    )

    confidence = lateral_score * 100

    is_lateral_pinch = (
        thumb_near_index_side
        and fingertips_not_too_close
        and index_position_ok
        and confidence >= 60
    )

    return is_lateral_pinch, confidence





def detect_multidigit_precision_grasp(
    finger_angles,
    thumb_fingertip_distances
):
    """
    Detect a multidigit precision grasp.

    Initial rule-based heuristic.
    Thresholds will be calibrated later.
    """

    participating_digits = sum(
        distance < 0.75
        for distance in thumb_fingertip_distances.values()
    )

    valid_angles = [
        angle
        for angle in finger_angles.values()
        if angle is not None
    ]

    if len(valid_angles) != 4:
        return False, 0.0

    moderately_flexed_fingers = sum(
        angle < 165
        for angle in valid_angles
    )

    multiple_digits_involved = (
        participating_digits >= 2
    )

    not_full_power_closure = (
        sum(
            angle < 100
            for angle in valid_angles
        ) <= 2
    )

    digit_score = min(
        participating_digits / 3.0,
        1.0
    )

    flexion_score = (
        moderately_flexed_fingers / 4.0
    )

    power_separation_score = (
        1.0 if not_full_power_closure else 0.0
    )

    precision_score = (
        0.50 * digit_score
        + 0.30 * flexion_score
        + 0.20 * power_separation_score
    )

    confidence = precision_score * 100

    is_multidigit_precision = (
        multiple_digits_involved
        and not_full_power_closure
        and confidence >= 60
    )

    return is_multidigit_precision, confidence