def create_frame_record(
    frame_number,
    timestamp_ms,
    session_id,
    trial_id,
    repetition_number,
    task_id,
    task_name,
    hand_label,
    hand_confidence,
    hand_center,
    grasp_detected,
    interaction_state,
    object_outcome,
    detected_object=None,
    object_distance=None,
    object_movement=0.0,
    fingertip_count=0,
    interaction_event=None,
    grasp_duration_ms=None,
    detected_grasp_type=None,
    detected_grasp_types=None,
    grasp_appropriateness=None,
    raw_grasp_type=None,
    grasp_family=None,
    family_appropriateness=None,
    best_guess=None,
    association_status=None,
    association_diagnostics=None,
):

    record = {
        "frame_number": frame_number,
        "timestamp_ms": timestamp_ms,

        "task": {
            "task_id": task_id,
            "task_name": task_name,
        },

        "hand": {
            "handedness": hand_label,
            "hand_confidence": hand_confidence,
            "hand_center": hand_center,
            "grasp_detected": grasp_detected,
            "detected_grasp_type": detected_grasp_type,
            "detected_grasp_types": detected_grasp_types,
            "grasp_appropriateness": grasp_appropriateness,
            "raw_grasp_type": raw_grasp_type,
            "grasp_family": grasp_family,
            "family_appropriateness": family_appropriateness,
            "best_guess": best_guess,
        },

        "interaction": {
            "state": interaction_state,
            "fingertip_count": fingertip_count,
            "object_distance": object_distance,
            "object_movement": object_movement,
            "association": {
                "status": association_status,
                "diagnostics": association_diagnostics,
            },
            "object_outcome": object_outcome,
            "event": interaction_event,
            "grasp_duration_ms": grasp_duration_ms,
        },

        "trial": {
            "session_id": session_id,
            "trial_id": trial_id,
            "repetition_number": repetition_number,
        },

        "object": None,

        
    }

    if detected_object is not None:
        record["object"] = {
            "class_name": detected_object["class_name"],
            "confidence": detected_object["confidence"],
            "bounding_box": detected_object["bounding_box"],
            "center": detected_object["center"],
            "track_id": detected_object.get("track_id"),
            "tracking_status": detected_object.get("tracking_status"),
            "object_id": detected_object.get("object_id"),
            "identity_status": detected_object.get("identity_status"),
            "identity_confidence": detected_object.get("identity_confidence"),
        }

    return record