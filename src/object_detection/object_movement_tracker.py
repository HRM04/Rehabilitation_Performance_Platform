import math


class ObjectMovementTracker:
    """
    Calculate movement separately for each application-level object ID.

    Call update() once per captured frame, with all objects
    detected in that frame.
    """

    def __init__(self):
        self.previous_positions = {}

    def update(self, detected_objects):
        movements = {}

        for obj in detected_objects:
            object_id = obj.get("object_id")

            # A missing ID cannot establish object continuity.
            if object_id is None:
                continue

            current_center = obj["center"]
            previous_center = self.previous_positions.get(object_id)

            if previous_center is None:
                movement = 0.0
            else:
                movement = math.dist(
                    previous_center,
                    current_center
                )

            movements[object_id] = movement
            self.previous_positions[object_id] = current_center

        return movements