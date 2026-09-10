import cv2


AREA_THRESHOLD = 850


def detect_objects(frame):
    """
    Detect objects in a camera frame using contour-based detection.

    Returns:
        detected_objects: list of dictionaries containing
        object center, bounding box, width, height, and orientation
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_normalized = cv2.normalize(
        gray,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    blurred = cv2.GaussianBlur(
        gray_normalized,
        (5, 5),
        0
    )

    filtered = cv2.bilateralFilter(
        blurred,
        10,
        55,
        55
    )

    mask = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        5
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_objects = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < AREA_THRESHOLD:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        center_x = int(
            moments["m10"] / moments["m00"]
        )

        center_y = int(
            moments["m01"] / moments["m00"]
        )

        x, y, width, height = cv2.boundingRect(contour)

        rotated_rect = cv2.minAreaRect(contour)

        angle = rotated_rect[2]

        detected_objects.append(
            {
                "center": (center_x, center_y),
                "bounding_box": (x, y, width, height),
                "width_px": width,
                "height_px": height,
                "orientation": angle,
                "area_px": area
            }
        )

    return detected_objects





def draw_detected_objects(frame, detected_objects):
    """
    Draw detected object information onto the frame.
    """

    for object_index, detected_object in enumerate(
        detected_objects
    ):

        x, y, width, height = (
            detected_object["bounding_box"]
        )

        center_x, center_y = (
            detected_object["center"]
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Object {object_index + 1}",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{width} x {height} px",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame