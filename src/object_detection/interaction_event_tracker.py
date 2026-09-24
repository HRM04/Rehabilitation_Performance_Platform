class InteractionEventTracker:
    """
    Track temporal interaction events separately for each
    hand-object pair.

    Events:
        GRASP
        REGRASP
        MANIPULATION
        TRANSPORT
        RELEASE

    Movement thresholds are provisional and should be
    validated later.
    """

    def __init__(
        self,
        manipulation_threshold=2.0,
        transport_threshold=10.0,
    ):
        self.manipulation_threshold = manipulation_threshold
        self.transport_threshold = transport_threshold

        self.pair_history = {}

    def update(
        self,
        hand_label,
        object_id,
        interaction_state,
        object_movement=0.0,
        timestamp_ms=None,
    ):
        """
        Update one hand-object pair.

        Returns:
            event_type or None
        """

        if object_id is None:
            return None

        pair_key = (hand_label, object_id)

        if pair_key not in self.pair_history:
            self.pair_history[pair_key] = {
                "previous_state": "NONE",
                "has_grasped_before": False,
                "last_event": None,
                "last_timestamp_ms": None,
                "movement_state": "NONE",
                "grasp_start_ms": None,
                "last_grasp_duration_ms": None,
            }

        history = self.pair_history[pair_key]

        previous_state = history["previous_state"]

        event_type = None

        history["last_grasp_duration_ms"] = None

        # -----------------------------------------
        # GRASP / REGRASP
        # -----------------------------------------
        if (
            previous_state != "GRASPING"
            and interaction_state == "GRASPING"
        ):
            history["grasp_start_ms"] = timestamp_ms

            if history["has_grasped_before"]:
                event_type = "REGRASP"
            else:
                event_type = "GRASP"
                history["has_grasped_before"] = True

        # -----------------------------------------
        # RELEASE
        # -----------------------------------------
        elif (
            previous_state == "GRASPING"
            and interaction_state != "GRASPING"
        ):
            event_type = "RELEASE"

            grasp_start_ms = history["grasp_start_ms"]

            if timestamp_ms is not None and grasp_start_ms is not None:
                history["last_grasp_duration_ms"] = max(
                    0,
                    timestamp_ms - grasp_start_ms,
                )

            history["grasp_start_ms"] = None

        # -----------------------------------------
        # MOVEMENT WHILE GRASPING
        # -----------------------------------------
        elif interaction_state == "GRASPING":

            if object_movement >= self.transport_threshold:
                new_movement_state = "TRANSPORT"

            elif object_movement >= self.manipulation_threshold:
                new_movement_state = "MANIPULATION"

            else:
                new_movement_state = "HOLDING"

            # Record an event only when the movement category changes.
            if new_movement_state != history["movement_state"]:

                if new_movement_state in (
                    "HOLDING",
                    "MANIPULATION",
                    "TRANSPORT",
                ):
                    event_type = new_movement_state

                history["movement_state"] = new_movement_state

                # Reset movement state when grasping ends.
        if interaction_state != "GRASPING":
            history["movement_state"] = "NONE"

        # Save the latest interaction state and event.
        history["previous_state"] = interaction_state
        history["last_event"] = event_type
        history["last_timestamp_ms"] = timestamp_ms

        return event_type




    def mark_unobserved(self, hand_label):
        """
        Clear active event histories for a hand when its
        hand-object interaction cannot be observed.

        Missing detection is not evidence of physical release.
        """

        keys_to_remove = [
            pair_key
            for pair_key in self.pair_history
            if pair_key[0] == hand_label
        ]

        for pair_key in keys_to_remove:
            del self.pair_history[pair_key]