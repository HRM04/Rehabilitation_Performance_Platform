"""Compensation assessment model for Delivery Dash.

Current mode: keyboard-controlled demonstration.
Events are simulated, NOT detected from patient movement.

Reference:
Lin et al. (2019), DOI: 10.1177/2055668319831631
"""

from dataclasses import dataclass, field


COMPENSATION_TYPES = {
    "shoulder_elevation": {
        "name": "Shoulder Elevation",
        "feedback": "Try to keep your shoulder relaxed.",
    },
    "trunk_rotation": {
        "name": "Trunk Rotation",
        "feedback": "Try to keep your body facing forward.",
    },
    "trunk_flexion": {
        "name": "Trunk Flexion",
        "feedback": "Try to reduce forward leaning.",
    },
    "slouching": {
        "name": "Slouching / Hip Extension",
        "feedback": "Check your sitting posture.",
    },
}

# Demonstration-only exercise classification criterion.
DEMO_FLAG_THRESHOLD = 0.05


@dataclass
class CompensationTracker:
    """Tracks compensation indicators across assessed frames."""

    total_frames: int = 0
    affected_frames: int = 0
    counts: dict = field(default_factory=lambda: {
        name: 0 for name in COMPENSATION_TYPES
    })
    demonstration_mode: bool = True

    def record_frame(self, active_compensations=()):
        """Record one frame with zero or more compensation indicators."""
        active = set(active_compensations)

        unknown = active - COMPENSATION_TYPES.keys()
        if unknown:
            raise ValueError(
                f"Unknown compensation types: {sorted(unknown)}"
            )

        self.total_frames += 1

        if active:
            self.affected_frames += 1

        for name in active:
            self.counts[name] += 1

    def summary(self):
        """Return frame-based compensation statistics."""
        total = self.total_frames

        rates = {
            name: round(100 * count / total, 1)
            if total else 0.0
            for name, count in self.counts.items()
        }

        return {
            "source": (
                "simulated"
                if self.demonstration_mode
                else "measured"
            ),
            "total_frames": total,
            "affected_frames": self.affected_frames,
            "overall_rate": (
                round(100 * self.affected_frames / total, 1)
                if total else 0.0
            ),
            "rates": rates,
            "flagged_types": [
                name for name, count in self.counts.items()
                if total and count / total >= DEMO_FLAG_THRESHOLD
            ],
        }

    def reset(self):
        """Start a new repetition."""
        self.total_frames = 0
        self.affected_frames = 0
        self.counts = {
            name: 0 for name in COMPENSATION_TYPES
        }


def feedback_for(compensation_type):
    """Return a patient-friendly demonstration feedback message."""
    return COMPENSATION_TYPES[compensation_type]["feedback"]
