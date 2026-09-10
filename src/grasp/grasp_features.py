import math

def calculate_thumb_index_distance(hand_data):
    thumb_tip = hand_data["THUMB_TIP"]
    index_tip = hand_data["INDEX_FINGER_TIP"]

    x_distance = (
        thumb_tip["x_scaled"] - index_tip["x_scaled"]
    )

    y_distance = (
        thumb_tip["y_scaled"] - index_tip["y_scaled"]
    )

    thumb_index_distance = (
        x_distance ** 2 + y_distance ** 2
    ) ** 0.5

    return thumb_index_distance




def calculate_joint_angle(point_a, point_b, point_c):
    """
    Calculate the 2D angle at point_b formed by:
    point_a -> point_b -> point_c
    """

    vector_ba = (
        point_a["x_scaled"] - point_b["x_scaled"],
        point_a["y_scaled"] - point_b["y_scaled"]
    )

    vector_bc = (
        point_c["x_scaled"] - point_b["x_scaled"],
        point_c["y_scaled"] - point_b["y_scaled"]
    )

    dot_product = (
        vector_ba[0] * vector_bc[0]
        + vector_ba[1] * vector_bc[1]
    )

    magnitude_ba = math.sqrt(
        vector_ba[0] ** 2 + vector_ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        vector_bc[0] ** 2 + vector_bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle_radians = math.acos(cosine_angle)

    return math.degrees(angle_radians)



def calculate_finger_angles(hand_data):
    """
    Calculate the PIP joint angle for the four fingers.

    Larger angle -> straighter finger
    Smaller angle -> more flexed finger
    """

    finger_landmarks = {
        "index": (
            "INDEX_FINGER_MCP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_DIP"
        ),
        "middle": (
            "MIDDLE_FINGER_MCP",
            "MIDDLE_FINGER_PIP",
            "MIDDLE_FINGER_DIP"
        ),
        "ring": (
            "RING_FINGER_MCP",
            "RING_FINGER_PIP",
            "RING_FINGER_DIP"
        ),
        "little": (
            "PINKY_MCP",
            "PINKY_PIP",
            "PINKY_DIP"
        )
    }

    finger_angles = {}

    for finger_name, landmarks in finger_landmarks.items():

        mcp_name, pip_name, dip_name = landmarks

        angle = calculate_joint_angle(
            hand_data[mcp_name],
            hand_data[pip_name],
            hand_data[dip_name]
        )

        finger_angles[finger_name] = angle

    return finger_angles


def calculate_thumb_opposition(hand_data):
    """
    Measure the thumb's distance from the index and middle fingers.

    Smaller distances generally indicate stronger thumb opposition.
    """

    thumb_tip = hand_data["THUMB_TIP"]
    index_tip = hand_data["INDEX_FINGER_TIP"]
    middle_tip = hand_data["MIDDLE_FINGER_TIP"]

    thumb_index_distance = (
        (
            thumb_tip["x_scaled"] - index_tip["x_scaled"]
        ) ** 2
        +
        (
            thumb_tip["y_scaled"] - index_tip["y_scaled"]
        ) ** 2
    ) ** 0.5

    thumb_middle_distance = (
        (
            thumb_tip["x_scaled"] - middle_tip["x_scaled"]
        ) ** 2
        +
        (
            thumb_tip["y_scaled"] - middle_tip["y_scaled"]
        ) ** 2
    ) ** 0.5

    return {
        "thumb_index": thumb_index_distance,
        "thumb_middle": thumb_middle_distance
    }



def calculate_finger_spread(hand_data):
    """
    Measure spacing between adjacent fingertips.

    Larger values generally indicate a more spread hand.
    """

    fingertip_pairs = {
        "index_middle": (
            "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_TIP"
        ),
        "middle_ring": (
            "MIDDLE_FINGER_TIP",
            "RING_FINGER_TIP"
        ),
        "ring_little": (
            "RING_FINGER_TIP",
            "PINKY_TIP"
        )
    }

    spread_distances = {}

    for pair_name, landmarks in fingertip_pairs.items():

        point_a = hand_data[landmarks[0]]
        point_b = hand_data[landmarks[1]]

        distance = (
            (
                point_a["x_scaled"] - point_b["x_scaled"]
            ) ** 2
            +
            (
                point_a["y_scaled"] - point_b["y_scaled"]
            ) ** 2
        ) ** 0.5

        spread_distances[pair_name] = distance

    return spread_distances




def calculate_hand_aperture(hand_data):
    """
    Estimate overall hand aperture using the distance
    between the thumb tip and little-finger tip.
    """

    thumb_tip = hand_data["THUMB_TIP"]
    little_tip = hand_data["PINKY_TIP"]

    x_distance = (
        thumb_tip["x_scaled"] - little_tip["x_scaled"]
    )

    y_distance = (
        thumb_tip["y_scaled"] - little_tip["y_scaled"]
    )

    return (
        x_distance ** 2
        + y_distance ** 2
    ) ** 0.5




def calculate_thumb_index_side_distance(hand_data):
    """
    Estimate how close the thumb is to the side of the index finger.

    We use the index PIP region as a simple landmark-based
    approximation of the lateral index-finger surface.
    """

    thumb_tip = hand_data["THUMB_TIP"]
    index_pip = hand_data["INDEX_FINGER_PIP"]

    x_distance = (
        thumb_tip["x_scaled"] - index_pip["x_scaled"]
    )

    y_distance = (
        thumb_tip["y_scaled"] - index_pip["y_scaled"]
    )

    return (
        x_distance ** 2
        + y_distance ** 2
    ) ** 0.5




def calculate_thumb_fingertip_distances(hand_data):
    """
    Measure thumb-tip distance to each fingertip.
    """

    thumb_tip = hand_data["THUMB_TIP"]

    fingertip_names = {
        "index": "INDEX_FINGER_TIP",
        "middle": "MIDDLE_FINGER_TIP",
        "ring": "RING_FINGER_TIP",
        "little": "PINKY_TIP"
    }

    distances = {}

    for finger_name, landmark_name in fingertip_names.items():

        fingertip = hand_data[landmark_name]

        x_distance = (
            thumb_tip["x_scaled"] - fingertip["x_scaled"]
        )

        y_distance = (
            thumb_tip["y_scaled"] - fingertip["y_scaled"]
        )

        distance = (
            x_distance ** 2
            + y_distance ** 2
        ) ** 0.5

        distances[finger_name] = distance

    return distances