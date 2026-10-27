from src.assessment.posture_calibration_session import (
    prepare_posture_calibration,
)


class PostureCalibrationController:
    """
    Manage a dedicated pre-task posture calibration.

    Calibration frames must be supplied separately from
    rehabilitation task frames.
    """

    def __init__(self, min_valid_frames=10):
        self.min_valid_frames = min_valid_frames
        self.status = "IDLE"
        self.frames = []
        self.result = None

    def start(self):
        """Begin a new calibration recording."""
        self.frames = []
        self.result = None
        self.status = "RECORDING"

    def add_frame(self, timestamp_ms, landmarks):
        """Collect a pose observation during calibration."""

        if self.status != "RECORDING":
            return False

        self.frames.append({
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
        })

        return True

    def finish(self):
        """Validate the collected calibration recording."""

        if self.status != "RECORDING":
            return self.result

        self.result = prepare_posture_calibration(
            self.frames,
            min_valid_frames=self.min_valid_frames,
        )

        if self.result["status"] == "READY":
            self.status = "READY"
        else:
            self.status = "RETRY_REQUIRED"

        return self.result

    def get_reference(self):
        """Return the reference only when calibration passed."""

        if self.status != "READY":
            return None

        return self.result["reference_posture"]
