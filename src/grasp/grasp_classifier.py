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