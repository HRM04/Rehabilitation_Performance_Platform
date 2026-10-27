"""Keyboard-controlled compensation demonstration.

This module simulates compensation indicators.
It does not measure patient posture or movement.
"""

from src.game_platform.compensation import (
    COMPENSATION_TYPES,
    feedback_for,
)


KEY_TO_COMPENSATION = {
    ord("5"): "shoulder_elevation",
    ord("6"): "trunk_rotation",
    ord("7"): "trunk_flexion",
    ord("8"): "slouching",
}


class CompensationDemo:
    def __init__(self):
        self.active = None

    def handle_key(self, key):
        """Select or clear a simulated compensation scenario."""
        if key == ord("0"):
            self.active = None
            return True

        if key in KEY_TO_COMPENSATION:
            self.active = KEY_TO_COMPENSATION[key]
            return True

        return False

    def current_indicators(self):
        """Return indicators for the current demonstration frame."""
        return [self.active] if self.active else []

    def current_feedback(self):
        """Return patient-facing feedback for the selected scenario."""
        if self.active is None:
            return "Keep going! Focus on controlled movement."

        return feedback_for(self.active)

    def current_label(self):
        if self.active is None:
            return "No simulated compensation"

        return COMPENSATION_TYPES[self.active]["name"]
