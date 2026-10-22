def create_repetition_record(
    repetition_number,
    start_frame,
    end_frame,
    start_timestamp_ms,
    end_timestamp_ms,
    total_object_movement,
    object_id=None,
):
    """Create one detected repetition segment."""

    return {
        "repetition_number": repetition_number,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_timestamp_ms": start_timestamp_ms,
        "end_timestamp_ms": end_timestamp_ms,
        "duration_ms": end_timestamp_ms - start_timestamp_ms,
        "total_object_movement": total_object_movement,
        "object_id": object_id,
    }


def segment_repetitions(
    rows,
    start_confirmation_frames=3,
    end_confirmation_frames=5,
    minimum_duration_ms=500,
    minimum_total_movement=20.0,
):
    """
    Detect bottle-transport repetition segments from sustained
    hand-object interaction rather than noisy interaction events.

    Start:
        A valid bottle association while the interaction state is
        GRASPING for several consecutive observations.

    Continue:
        The interaction remains in GRASPING, HOLDING,
        MANIPULATION, or TRANSPORT.
        Short observation losses are tolerated.

    End:
        Interaction is absent for several consecutive observations.

    A candidate is retained only when it has sufficient duration
    and observed object movement.
    """

    repetitions = []

    candidate_start_count = 0
    missing_count = 0

    in_repetition = False

    start_frame = None
    start_timestamp_ms = None

    last_active_frame = None
    last_active_timestamp_ms = None

    total_object_movement = 0.0

    repetition_number = 1

    active_object_id = None

    for row in rows:

        frame_number = int(row["frame_number"])
        timestamp_ms = int(row["timestamp_ms"])

        interaction_state = row.get("interaction_state")
        object_class = row.get("object_class")

        movement_text = row.get("object_movement")

        try:
            object_movement = float(movement_text)
        except (TypeError, ValueError):
            object_movement = 0.0

        # For the current bottle-transport task, interaction
        # evidence requires a detected bottle.
        valid_object = object_class == "bottle"

        # GRASPING is our strongest current evidence that the
        # hand is actively interacting with the object.
        active_states = {
            "GRASPING",
            "HOLDING",
            "MANIPULATION",
            "TRANSPORT",
        }

        active_interaction = (
            valid_object
            and interaction_state in active_states
        )


        if in_repetition and active_object_id is not None:
            active_interaction = (
                active_interaction
                and row.get("object_id") == active_object_id
            )

        # -------------------------------------------------
        # Waiting for a repetition to begin
        # -------------------------------------------------
                
        
        if not in_repetition:

            if (
                valid_object
                and interaction_state == "GRASPING"
            ):

                current_object_id = row.get("object_id")

                if (
                    candidate_start_count > 0
                    and current_object_id != active_object_id
                ):
                    candidate_start_count = 0
                    total_object_movement = 0.0

                candidate_start_count += 1

                if candidate_start_count == 1:
                    candidate_start_frame = frame_number
                    candidate_start_timestamp_ms = timestamp_ms
                    total_object_movement = 0.0
                    active_object_id = row.get("object_id")

                # Include movement from every confirmation frame.
                total_object_movement += object_movement

                if candidate_start_count >= start_confirmation_frames:

                    in_repetition = True

                    start_frame = candidate_start_frame
                    start_timestamp_ms = candidate_start_timestamp_ms

                    last_active_frame = frame_number
                    last_active_timestamp_ms = timestamp_ms

                    missing_count = 0

            else:
                candidate_start_count = 0
                total_object_movement = 0.0

            continue



        # -------------------------------------------------
        # Repetition currently active
        # -------------------------------------------------
        if active_interaction:

            missing_count = 0

            last_active_frame = frame_number
            last_active_timestamp_ms = timestamp_ms

            total_object_movement += object_movement

            continue

        # Interaction temporarily disappeared.
        missing_count += 1

        # Do not immediately end the repetition.
        if missing_count < end_confirmation_frames:
            continue

        # -------------------------------------------------
        # Confirm repetition end
        # -------------------------------------------------
        duration_ms = (
            last_active_timestamp_ms
            - start_timestamp_ms
        )

        if (
            duration_ms >= minimum_duration_ms
            and total_object_movement >= minimum_total_movement
        ):
            repetitions.append(
                create_repetition_record(
                    repetition_number=repetition_number,
                    start_frame=start_frame,
                    end_frame=last_active_frame,
                    start_timestamp_ms=start_timestamp_ms,
                    end_timestamp_ms=last_active_timestamp_ms,
                    total_object_movement=total_object_movement,
                    object_id=active_object_id,
                )
            )

            repetition_number += 1

        # Reset for the next candidate.
        in_repetition = False

        candidate_start_count = 0
        missing_count = 0

        start_frame = None
        start_timestamp_ms = None

        last_active_frame = None
        last_active_timestamp_ms = None

        total_object_movement = 0.0
        active_object_id = None

    # -------------------------------------------------
    # Recording ended during an active repetition
    # -------------------------------------------------
    if in_repetition and last_active_timestamp_ms is not None:

        duration_ms = (
            last_active_timestamp_ms
            - start_timestamp_ms
        )

        if (
            duration_ms >= minimum_duration_ms
            and total_object_movement >= minimum_total_movement
        ):
            repetition = create_repetition_record(
                repetition_number=repetition_number,
                start_frame=start_frame,
                end_frame=last_active_frame,
                start_timestamp_ms=start_timestamp_ms,
                end_timestamp_ms=last_active_timestamp_ms,
                total_object_movement=total_object_movement,
            )

            repetition["ended_with_recording"] = True

            repetitions.append(repetition)

    return repetitions