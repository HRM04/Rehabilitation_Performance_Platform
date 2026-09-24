class ObjectDetectionTracker:

    def __init__(
        self,
        lost_after_frames=3,
        uncertain_threshold=0.25
    ):
        self.lost_after_frames = lost_after_frames
        self.uncertain_threshold = uncertain_threshold

        self.object_seen_before = False
        self.missing_frame_count = 0


    def update(self, permitted_objects):

        # No permitted object detected in this frame
        if not permitted_objects:

            if self.object_seen_before:
                self.missing_frame_count += 1

                if self.missing_frame_count >= self.lost_after_frames:
                    return "LOST_DETECTION"

                return "TEMPORARILY_MISSING"

            return "NO_PERMITTED_OBJECT"

        # At least one permitted object is visible
        self.object_seen_before = True
        self.missing_frame_count = 0

        highest_confidence = max(
            obj["confidence"]
            for obj in permitted_objects
        )

        if highest_confidence < self.uncertain_threshold:
            return "UNCERTAIN"

        return "DETECTED"