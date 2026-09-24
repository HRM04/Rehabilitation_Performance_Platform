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
        },

        "interaction": {
            "state": interaction_state,
            "fingertip_count": fingertip_count,
            "object_distance": object_distance,
            "object_movement": object_movement,
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