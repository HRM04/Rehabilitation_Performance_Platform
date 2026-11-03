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



PINCH_STATS = {"errors": 0}


def pinch_contact_at_object(hand_data, detected_object, fingertip_count, family, margin=20):
    """True when the thumb and at least one other fingertip are on the object, two or more
    fingertips touch it, and the hand shape reads as a precision (fingertip) grasp.
    Lets a pinch on a large object reach GRASPING when no exact grasp detector fires."""
    if detected_object is None or family != "PRECISION" or (fingertip_count or 0) < 2:
        return False
    x1, y1, x2, y2 = detected_object["bounding_box"]

    def inside(name):
        p = hand_data[name]
        return x1 - margin <= p["x_px"] <= x2 + margin and y1 - margin <= p["y_px"] <= y2 + margin

    others = ("INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP")
    return inside("THUMB_TIP") and any(inside(n) for n in others)


def note_pinch_error(error):
    PINCH_STATS["errors"] += 1
    if PINCH_STATS["errors"] == 1:
        print("[PINCH STATE] check unavailable:", repr(error))


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

    def reset(self):
        """
        Clear the stable state and any pending state transition.
        Call this when a hand has been missing for too long.
        """
        self.current_state = "NONE"
        self.candidate_state = "NONE"
        self.candidate_count = 0

def distance_to_object_box(point, detected_object):
    """Pixel distance from a point to an object's bounding box."""
    x, y = point
    x1, y1, x2, y2 = detected_object["bounding_box"]

    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)

    return math.hypot(dx, dy)


def select_hand_object_association(
    hand_center,
    hand_data,
    detected_objects,
    previous_object_id=None,
    max_distance=50,
    ambiguity_margin=5,
):
    """
    Associate a hand with an observed object using geometric evidence.

    Priority:
      1. Plausible object proximity
      2. Fingertip evidence
      3. Object continuity
      4. Bounding-box distance

    Returns:
        (object, "ASSOCIATED", diagnostics)
        (None, "AMBIGUOUS", diagnostics)
        (None, "NONE", diagnostics)

    Scores and thresholds are engineering heuristics, not
    calibrated probabilities or proof of physical contact.
    """
    candidates = []
    objects_by_id = {}

    for detected_object in detected_objects:
        object_id = detected_object.get("object_id")

        if object_id is None:
            continue

        objects_by_id.setdefault(object_id, []).append(
            detected_object
        )

    # Conflicting detections sharing one identity cannot be
    # safely interpreted as independent physical objects.
    conflicting_ids = set()

    for object_id, observations in objects_by_id.items():
        if len(observations) <= 1:
            continue

        reference_box = observations[0]["bounding_box"]

        if any(
            item["bounding_box"] != reference_box
            for item in observations[1:]
        ):
            conflicting_ids.add(object_id)

    for object_id, observations in objects_by_id.items():
        # Identical duplicate observations are counted once.
        detected_object = observations[0]

        distance = distance_to_object_box(
            hand_center,
            detected_object,
        )

        fingertips = count_fingertips_near_object(
            hand_data,
            detected_object,
        )

        if distance > max_distance and fingertips < 2:
            continue

        candidates.append({
            "object": detected_object,
            "object_id": object_id,
            "distance_px": distance,
            "fingertips": fingertips,
            "conflicting_identity": (
                object_id in conflicting_ids
            ),
        })

    diagnostics = {
        "candidates": [
            {
                "object_id": item["object_id"],
                "distance_px": item["distance_px"],
                "fingertips": item["fingertips"],
                "conflicting_identity": (
                    item["conflicting_identity"]
                ),
            }
            for item in candidates
        ]
    }

    if not candidates:
        return None, "NONE", diagnostics

    candidates.sort(
        key=lambda item: (
            -item["fingertips"],
            item["distance_px"],
        )
    )

    best = candidates[0]

    if len(candidates) == 1:
        if best["conflicting_identity"]:
            return None, "AMBIGUOUS", diagnostics

        return best["object"], "ASSOCIATED", diagnostics

    second = candidates[1]

    fingertip_difference = (
        best["fingertips"] - second["fingertips"]
    )

    # Strong fingertip evidence overrides hand-centre proximity.
    if (
        best["fingertips"] >= 2
        and fingertip_difference >= 2
    ):
        if best["conflicting_identity"]:
            return None, "AMBIGUOUS", diagnostics

        return best["object"], "ASSOCIATED", diagnostics

    # If competing objects both have fingertip evidence,
    # avoid an unsupported association unless continuity
    # can distinguish them and the evidence is not strongly
    # against the previously associated object.
    if (
        best["fingertips"] >= 2
        and second["fingertips"] >= 2
    ):
        # Continuity can resolve an exact fingertip-evidence tie,
        # but cannot override stronger evidence for another object.
        if best["fingertips"] == second["fingertips"]:
            tied_candidates = [
                item for item in candidates
                if item["fingertips"] == best["fingertips"]
            ]

            previous_matches = [
                item for item in tied_candidates
                if item["object_id"] == previous_object_id
                and not item["conflicting_identity"]
                and item["distance_px"] <= max_distance
            ]

            if len(previous_matches) == 1:
                diagnostics["resolution"] = "CONTINUITY_TIE_BREAK"
                return (
                    previous_matches[0]["object"],
                    "ASSOCIATED",
                    diagnostics,
                )

        return None, "AMBIGUOUS", diagnostics

    # Previous association can resolve otherwise weak evidence.
    previous_candidates = [
        item for item in candidates
        if item["object_id"] == previous_object_id
        and not item["conflicting_identity"]
    ]

    if previous_candidates:
        previous = previous_candidates[0]

        if (
            best["fingertips"] - previous["fingertips"] <= 1
            and previous["distance_px"] <= max_distance
        ):
            return previous["object"], "ASSOCIATED", diagnostics

    # Resolve ties using geometric proximity only when the
    # distance advantage is sufficiently clear.
    if best["fingertips"] == second["fingertips"]:
        distance_difference = (
            second["distance_px"] - best["distance_px"]
        )

        if distance_difference <= ambiguity_margin:
            return None, "AMBIGUOUS", diagnostics

    if (
        best["distance_px"] + 15
        < second["distance_px"]
    ):
        if best["conflicting_identity"]:
            return None, "AMBIGUOUS", diagnostics

        return best["object"], "ASSOCIATED", diagnostics

    return None, "AMBIGUOUS", diagnostics
