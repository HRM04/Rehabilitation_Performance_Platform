"""Configuration options for the Delivery Dash demonstration."""

from dataclasses import dataclass

from src.platform_config.object_library import OBJECT_LIBRARY


TRAJECTORIES = {
    "horizontal": "Horizontal",
    "vertical": "Vertical",
    "diagonal": "Diagonal",
    "curved": "Curved",
    "s_shaped": "S-Shaped",
}

REPETITION_OPTIONS = (3, 5, 10, 12)


@dataclass
class GameConfig:
    object_name: str = "bottle"
    trajectory: str = "horizontal"
    repetitions: int = 10

    def validate(self):
        if self.object_name not in OBJECT_LIBRARY:
            raise ValueError(
                f"Unsupported object: {self.object_name}"
            )

        if self.trajectory not in TRAJECTORIES:
            raise ValueError(
                f"Unsupported trajectory: {self.trajectory}"
            )

        if self.repetitions not in REPETITION_OPTIONS:
            raise ValueError(
                f"Unsupported repetition count: {self.repetitions}"
            )

        return self


def available_objects():
    return [
        {
            "id": object_id,
            "name": details["display_name"],
            "status": details["status"],
            "expected_grasps": details["expected_grasps"],
        }
        for object_id, details in OBJECT_LIBRARY.items()
    ]
