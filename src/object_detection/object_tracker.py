import math
from src.object_detection.duplicate_filter import calculate_iou

from src.object_detection.yolo_detector import model


def track_objects(frame, allowed_classes=None):
    """
    Detect and track objects in a camera frame.

    Parameters
    ----------
    frame:
        Camera frame to process.

    allowed_classes:
        Optional list of permitted class names.
        If None, track all supported object classes.

    Returns
    -------
    list:
        Detected objects, including tracking IDs when available.
    """

    class_ids = None

    if allowed_classes is not None:
        class_ids = [
            class_id
            for class_id, class_name in model.names.items()
            if class_name in allowed_classes
        ]

        if not class_ids:
            return []

    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        classes=class_ids,
        tracker="bytetrack.yaml",
        device="mps",
        verbose=False
    )

    tracked_objects = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            track_id = (
                int(box.id[0])
                if box.id is not None
                else None
            )

            tracked_objects.append({
                "track_id": track_id,
                "class_name": class_name,
                "confidence": confidence,
                "bounding_box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "tracking_status": (
                    "TRACKED"
                    if track_id is not None
                    else "UNASSIGNED"
                )
            })

    return tracked_objects


class ObjectIdentityTracker:
    """
    Maintain application-level IDs for detected objects.

    Association priority:
    1. Reuse the same application ID when the YOLO track ID
       matches a recently seen object of the same class.
    2. Otherwise, try to reassociate by spatial proximity.
    3. Otherwise, create a new application ID.

    This is still a prototype. Application IDs are not proof
    of true physical-object identity.
    """

    def __init__(
        self,
        max_missing_frames=15,
        max_match_distance=100
    ):
        self.max_missing_frames = max_missing_frames
        self.max_match_distance = max_match_distance

        self.frame_number = 0
        self.next_object_id = 1

        self.active_objects = {}

    def _is_expired(self, previous_object):
        missing_frames = (
            self.frame_number
            - previous_object["last_seen_frame"]
            - 1
        )
        return missing_frames > self.max_missing_frames

    def _distance(self, center_a, center_b):
        x1, y1 = center_a
        x2, y2 = center_b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def update(self, detected_objects):
        """
        Associate current detections with previous objects.

        Returns the current detections with:
            object_id
            identity_status
            identity_confidence
        """

        self.frame_number += 1

        # Remove expired objects
        expired_ids = []

        for object_id, previous in self.active_objects.items():
            if self._is_expired(previous):
                expired_ids.append(object_id)

        for object_id in expired_ids:
            del self.active_objects[object_id]

        current_objects = [
            detected_object.copy()
            for detected_object in detected_objects
        ]

        unmatched_previous_ids = set(self.active_objects.keys())
        assigned_current_indices = set()

        # -------------------------------------------------
        # PASS 1: Match by YOLO track ID continuity
        # -------------------------------------------------
        for index, detected_object in enumerate(current_objects):

            yolo_track_id = detected_object.get("track_id")

            if yolo_track_id is None:
                continue

            candidates = []

            for object_id in unmatched_previous_ids:
                previous = self.active_objects[object_id]

                if previous["class_name"] != detected_object["class_name"]:
                    continue

                if previous.get("yolo_track_id") == yolo_track_id:
                    candidates.append(object_id)

            if len(candidates) == 1:
                object_id = candidates[0]

                detected_object["object_id"] = object_id
                detected_object["identity_status"] = "TRACK_CONTINUED"
                detected_object["identity_confidence"] = "HIGH"

                assigned_current_indices.add(index)
                unmatched_previous_ids.remove(object_id)

        # -------------------------------------------------
        # PASS 2: Match remaining detections by distance
        # -------------------------------------------------
        for index, detected_object in enumerate(current_objects):

            if index in assigned_current_indices:
                continue

            spatial_candidates = []

            for object_id in unmatched_previous_ids:
                previous = self.active_objects[object_id]

                if previous["class_name"] != detected_object["class_name"]:
                    continue

                distance = self._distance(
                    detected_object["center"],
                    previous["center"]
                )

                if distance <= self.max_match_distance:
                    spatial_candidates.append(
                        (distance, object_id)
                    )

            spatial_candidates.sort(
                key=lambda item: item[0]
            )

            if len(spatial_candidates) == 1:
                distance, object_id = spatial_candidates[0]

                detected_object["object_id"] = object_id
                detected_object["identity_status"] = "REASSOCIATED"
                detected_object["identity_confidence"] = "PROVISIONAL"

                assigned_current_indices.add(index)
                unmatched_previous_ids.remove(object_id)

            elif len(spatial_candidates) > 1:
                # Resolve ambiguity only with strong geometric evidence.
                strong_matches = []

                for distance, candidate_id in spatial_candidates:
                    previous = self.active_objects[candidate_id]

                    overlap = calculate_iou(
                        detected_object["bounding_box"],
                        previous["bounding_box"]
                    )

                    missing_frames = (
                        self.frame_number
                        - previous["last_seen_frame"]
                        - 1
                    )

                    if overlap >= 0.85 and missing_frames <= 3:
                        strong_matches.append(
                            (overlap, -missing_frames, candidate_id)
                        )

                strong_matches.sort(reverse=True)

                # Require a clear overlap advantage over the runner-up.
                if (
                    strong_matches
                    and (
                        len(strong_matches) == 1
                        or strong_matches[0][0] - strong_matches[1][0] >= 0.05
                    )
                ):
                    object_id = strong_matches[0][2]

                    detected_object["object_id"] = object_id
                    detected_object["identity_status"] = "REASSOCIATED"
                    detected_object["identity_confidence"] = "PROVISIONAL"

                    unmatched_previous_ids.remove(object_id)
                else:
                    object_id = self.next_object_id
                    self.next_object_id += 1

                    detected_object["object_id"] = object_id
                    detected_object["identity_status"] = "AMBIGUOUS_NEW"
                    detected_object["identity_confidence"] = "UNCERTAIN"

                assigned_current_indices.add(index)

        # -------------------------------------------------
        # PASS 3: Assign brand-new IDs
        # -------------------------------------------------
        for index, detected_object in enumerate(current_objects):

            if index in assigned_current_indices:
                continue

            object_id = self.next_object_id
            self.next_object_id += 1

            detected_object["object_id"] = object_id
            detected_object["identity_status"] = "NEW"
            detected_object["identity_confidence"] = "UNVERIFIED"

        # -------------------------------------------------
        # Update active objects with current observations
        # -------------------------------------------------
        updated_objects = {}

        for detected_object in current_objects:
            object_id = detected_object["object_id"]

            updated_objects[object_id] = {
                "class_name": detected_object["class_name"],
                "center": detected_object["center"],
                "bounding_box": detected_object["bounding_box"],
                "last_seen_frame": self.frame_number,
                "yolo_track_id": detected_object.get("track_id"),
            }

        self.active_objects.update(updated_objects)

        return current_objects