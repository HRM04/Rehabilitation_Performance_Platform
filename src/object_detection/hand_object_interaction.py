import math


def find_nearest_object(hand_center, detected_objects):
    """
    Find the detected object closest to the hand center.
    """

    if not detected_objects:
        return None, None

    hand_x, hand_y = hand_center

    nearest_object = None
    nearest_distance = float("inf")

    for detected_object in detected_objects:

        object_x, object_y = detected_object["center"]

        distance = math.sqrt(
            (hand_x - object_x) ** 2
            + (hand_y - object_y) ** 2
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_object = detected_object

    return nearest_object, nearest_distance





def is_hand_near_object(
    hand_center,
    detected_object,
    margin=50
):
    """
    Check whether the hand center is inside or near
    the object's bounding box.
    """

    hand_x, hand_y = hand_center

    x1, y1, x2, y2 = detected_object["bounding_box"]

    return (
        x1 - margin <= hand_x <= x2 + margin
        and
        y1 - margin <= hand_y <= y2 + margin
    )



def count_fingertips_near_object(
    hand_data,
    detected_object,
    margin=20
):
    """
    Count how many fingertips are inside or near
    the object's bounding box.
    """

    fingertip_names = [
        "THUMB_TIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_TIP",
        "PINKY_TIP"
    ]

    x1, y1, x2, y2 = detected_object["bounding_box"]

    fingertip_count = 0

    for fingertip_name in fingertip_names:

        fingertip_x = hand_data[fingertip_name]["x_px"]
        fingertip_y = hand_data[fingertip_name]["y_px"]

        if (
            x1 - margin <= fingertip_x <= x2 + margin
            and
            y1 - margin <= fingertip_y <= y2 + margin
        ):
            fingertip_count += 1

    return fingertip_count





def determine_interaction_state(
    hand_near_object,
    fingertips_near_object,
    grasp_detected=False
):
    """
    Determine the current hand-object interaction state.

    Parameters
    ----------
    hand_near_object : bool
        Whether the hand is within the expanded object bounding box.

    fingertips_near_object : int
        Number of fingertips currently near/inside the object's
        interaction region.

    grasp_detected : bool
        Whether one or more grasp detectors currently indicate
        a grasp-like hand configuration.

    Returns
    -------
    str
        One of:
        "NONE"
        "NEAR"
        "CONTACT"
        "GRASPING"
    """

    if not hand_near_object:
        return "NONE"

    if fingertips_near_object == 0:
        return "NEAR"

    if fingertips_near_object >= 1:
        if fingertips_near_object >= 2 and grasp_detected:
            return "GRASPING"

        return "CONTACT"

    return "NONE"



def calculate_object_movement(
    previous_center,
    current_center
):
    if previous_center is None:
        return 0.0

    previous_x, previous_y = previous_center
    current_x, current_y = current_center

    movement = (
        (current_x - previous_x) ** 2
        + (current_y - previous_y) ** 2
    ) ** 0.5

    return movement




class InteractionStateTracker:

    def __init__(self, required_frames=3):
        self.required_frames = required_frames

        self.current_state = "NONE"
        self.candidate_state = "NONE"
        self.candidate_count = 0


    def update(self, new_candidate_state):

        # If the new candidate matches the previous candidate,
        # increase how long it has persisted.
        if new_candidate_state == self.candidate_state:
            self.candidate_count += 1

        else:
            # A different candidate appeared,
            # so start counting again.
            self.candidate_state = new_candidate_state
            self.candidate_count = 1


        # Only change the stable state if the candidate
        # has persisted long enough.
        if self.candidate_count >= self.required_frames:
            self.current_state = self.candidate_state


        return self.current_state