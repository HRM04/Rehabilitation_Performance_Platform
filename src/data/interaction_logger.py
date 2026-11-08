import csv


def create_interaction_logger(output_path):

    csv_file = open(
        output_path,
        "w",
        newline=""
    )

    fieldnames = [
        "session_id",
        "trial_id",
        "repetition_number",

        "frame_number",
        "timestamp_ms",

        "task_id",
        "task_name",

        "handedness",
        "hand_confidence",

        "hand_center_x",
        "hand_center_y",

        "grasp_detected",
        "detected_grasp_type",
        "detected_grasp_types",
        "grasp_appropriateness",
        "appropriateness_reason",
        "raw_grasp_type",
        "grasp_family",
        "family_appropriateness",
        "best_guess",

        "interaction_state",
    "association_status",
    "association_resolution",
    "association_candidate_count",
        "interaction_event",
        "grasp_duration_ms",
        "object_outcome",
        "fingertip_count",
        "object_distance",
        "object_movement",

        "object_class",
        "object_confidence",

        "track_id",
        "tracking_status",

        "object_id",
        "identity_status",
        "identity_confidence",

        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",

        "object_center_x",
        "object_center_y",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    return csv_file, writer


def _appropriateness_reason(hand, association_status, object_present):
    """Plain-language reason for the appropriateness outcome."""
    outcome = hand.get("grasp_appropriateness")
    grasp = hand.get("detected_grasp_type")
    if outcome == "NOT_ASSESSED":
        if association_status and association_status != "ASSOCIATED":
            return "NO_OBJECT_ASSOCIATED:" + str(association_status)
        if not object_present:
            return "NO_OBJECT_ASSOCIATED"
        return "OBJECT_NOT_NEAR_HAND_OR_NOT_TASK_OBJECT"
    if outcome == "UNCERTAIN":
        return "GRASP_" + str(grasp)
    if outcome == "APPROPRIATE":
        return "GRASP_IN_EXPECTED_SET"
    if outcome == "INAPPROPRIATE":
        return "GRASP_NOT_IN_EXPECTED_SET"
    return None


def log_frame_record(
    writer,
    frame_record
):

    hand = frame_record["hand"]
    interaction = frame_record["interaction"]
    association = interaction.get("association") or {}
    association_diagnostics = association.get("diagnostics") or {}
    task = frame_record["task"]
    detected_object = frame_record["object"]
    trial = frame_record["trial"]

    hand_center_x, hand_center_y = hand["hand_center"]

    row = {
        "frame_number": frame_record["frame_number"],
        "timestamp_ms": frame_record["timestamp_ms"],

        "session_id": trial["session_id"],
        "trial_id": trial["trial_id"],
        "repetition_number": trial["repetition_number"],

        "task_id": task["task_id"],
        "task_name": task["task_name"],

        "handedness": hand["handedness"],
        "hand_confidence": hand["hand_confidence"],

        "hand_center_x": hand_center_x,
        "hand_center_y": hand_center_y,

        "grasp_detected": hand["grasp_detected"],
        "detected_grasp_type": hand.get("detected_grasp_type"),
        "detected_grasp_types": ", ".join(
            hand.get("detected_grasp_types") or []
        ),
        "grasp_appropriateness": hand.get("grasp_appropriateness"),
        "raw_grasp_type": hand.get("raw_grasp_type"),
        "grasp_family": hand.get("grasp_family"),
        "family_appropriateness": hand.get("family_appropriateness"),
        "best_guess": hand.get("best_guess"),
        "appropriateness_reason": _appropriateness_reason(
            hand, association.get("status"), detected_object is not None),

        "interaction_state": interaction["state"],
        "association_status": association.get("status"),
        "association_resolution": association_diagnostics.get("resolution"),
        "association_candidate_count": len(
            association_diagnostics.get("candidates") or []
        ),
        "interaction_event": interaction.get("event"),
        "object_outcome": interaction["object_outcome"],
        "fingertip_count": interaction["fingertip_count"],
        "object_distance": interaction["object_distance"],
        "object_movement": interaction["object_movement"],

        "object_class": None,
        "object_confidence": None,

        "track_id": None,
        "tracking_status": None,

        "object_id": None,
        "identity_status": None,
        "identity_confidence": None,

        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,

        "object_center_x": None,
        "object_center_y": None,

        "interaction_event": interaction.get("event"),
        "grasp_duration_ms": interaction.get("grasp_duration_ms"),
        
    }

    if detected_object is not None:

        x1, y1, x2, y2 = detected_object["bounding_box"]
        object_center_x, object_center_y = detected_object["center"]

        row.update({
            "object_class": detected_object["class_name"],
            "object_confidence": detected_object["confidence"],

            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,

            "object_center_x": object_center_x,
            "object_center_y": object_center_y,

            "track_id": detected_object.get("track_id"),
            "tracking_status": detected_object.get("tracking_status"),

            "object_id": detected_object.get("object_id"),
            "identity_status": detected_object.get("identity_status"),
            "identity_confidence": detected_object.get("identity_confidence"),

        })

    writer.writerow(row)




def log_no_hand_interaction(
    writer,
    frame_number,
    timestamp_ms,
    session_id,
    trial_id,
    repetition_number,
    task_id,
    task_name,
    object_outcome
):
    """
    Record a frame where no hands were detected.

    Hand-specific and interaction measurements remain
    empty because no valid hand observation is available.
    """

    row = {
        "session_id": session_id,
        "trial_id": trial_id,
        "repetition_number": repetition_number,

        "frame_number": frame_number,
        "timestamp_ms": timestamp_ms,

        "task_id": task_id,
        "task_name": task_name,

        "handedness": None,
        "hand_confidence": None,

        "hand_center_x": None,
        "hand_center_y": None,

        "grasp_detected": None,
        "appropriateness_reason": "NO_HAND_DETECTED",

        "interaction_state": None,
        "association_status": None,
        "association_resolution": None,
        "association_candidate_count": 0,
        "interaction_event": None,
        "object_outcome": object_outcome,
        "fingertip_count": None,
        "object_distance": None,
        "object_movement": None,

        "object_class": None,
        "object_confidence": None,

        "track_id": None,
        "tracking_status": None,

        "object_id": None,
        "identity_status": None,
        "identity_confidence": None,

        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,

        "object_center_x": None,
        "object_center_y": None,

        "grasp_duration_ms": None,
    }

    writer.writerow(row)