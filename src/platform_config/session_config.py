"""
Session configuration for the unified
rehabilitation platform.
"""

from dataclasses import dataclass, field
from datetime import datetime

from src.platform_config.object_library import get_object


TRAJECTORY_PRESETS = (
    "horizontal",
    "vertical",
    "diagonal",
    "curved",
    "s_shaped",
)


@dataclass
class SessionConfig:
    object_id: str = "bottle"
    trajectory: str = "horizontal"
    patient_id: str = "demo_patient"
    session_id: str = field(
        default_factory=lambda:
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_repetitions: int = 3

    def validate(self):
        get_object(self.object_id)

        if self.trajectory not in TRAJECTORY_PRESETS:
            raise ValueError(
                f"Unsupported trajectory: {self.trajectory}"
            )

        if self.target_repetitions < 1:
            raise ValueError(
                "Target repetitions must be at least 1."
            )

        return True

    def get_task_details(self):
        obj = get_object(self.object_id)

        return {
            "task_id": f"{self.object_id.replace(' ', '_')}_transport",
            "task_name": f"{obj['display_name']} Transport",
            "permitted_objects": [self.object_id],
            "expected_grasps": obj["expected_grasps"],
            "trajectory": self.trajectory,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "target_repetitions": self.target_repetitions,
        }
