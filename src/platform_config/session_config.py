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
    session_purpose: str = "regular"
    volunteer_session: int = 1
    session_id: str = field(
        default_factory=lambda:
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_repetitions: int = 3

    # Optional assessment features
    trajectory_enabled: bool = True
    orientation_enabled: bool = False
    segmentation_enabled: bool = False

    def validate(self):
        if self.session_purpose not in ("regular", "testing"):
            raise ValueError("Invalid session purpose.")
        if self.volunteer_session < 1:
            raise ValueError("Volunteer session must be positive.")
        for name in (
            "trajectory_enabled",
            "orientation_enabled",
            "segmentation_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False.")

        if self.orientation_enabled and not self.trajectory_enabled:
            raise ValueError(
                "Orientation assessment requires trajectory mode."
            )

        if self.orientation_enabled and not self.segmentation_enabled:
            raise ValueError(
                "Orientation assessment requires segmentation."
            )

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
            "trajectory_enabled": self.trajectory_enabled,
            "orientation_enabled": self.orientation_enabled,
            "segmentation_enabled": self.segmentation_enabled,
        }
