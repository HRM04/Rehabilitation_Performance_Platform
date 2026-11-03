
from src.assessment.repetition_analysis import segment_repetitions


def make_row(frame, state, movement=0.0, object_class="bottle"):
    return {
        "frame_number": frame,
        "timestamp_ms": frame * 100,
        "interaction_state": state,
        "handedness": "Right",
        "object_class": object_class,
        "object_movement": movement,
    }


def add_frames(rows, start, count, state, movement=0.0,
               object_class="bottle"):
    for frame in range(start, start + count):
        rows.append(make_row(frame, state, movement, object_class))


# -------------------------------------------------
# TEST 4A: One valid repetition
# -------------------------------------------------

rows = []

add_frames(rows, 0, 3, "GRASPING", 5.0)
add_frames(rows, 3, 7, "TRANSPORT", 10.0)
add_frames(rows, 10, 5, "NONE")

repetitions = segment_repetitions(rows)

print("DEBUG - Detected repetitions:", repetitions)

assert len(repetitions) == 1
assert repetitions[0]["start_frame"] == 0
assert repetitions[0]["end_frame"] == 9
assert repetitions[0]["duration_ms"] == 900

print("PASS: Valid repetition detected.")


# -------------------------------------------------
# TEST 4B: Insufficient grasp confirmation
# -------------------------------------------------

rows = []

add_frames(rows, 0, 2, "GRASPING", 5.0)
add_frames(rows, 2, 8, "TRANSPORT", 10.0)
add_frames(rows, 10, 5, "NONE")

repetitions = segment_repetitions(rows)

assert len(repetitions) == 0

print("PASS: Unconfirmed grasp rejected.")


# -------------------------------------------------
# TEST 4C: Temporary interaction loss
# -------------------------------------------------

rows = []

add_frames(rows, 0, 3, "GRASPING", 5.0)
add_frames(rows, 3, 4, "TRANSPORT", 10.0)
add_frames(rows, 7, 3, "NONE")
add_frames(rows, 10, 4, "TRANSPORT", 10.0)
add_frames(rows, 14, 5, "NONE")

repetitions = segment_repetitions(rows)

assert len(repetitions) == 1
assert repetitions[0]["start_frame"] == 0
assert repetitions[0]["end_frame"] == 13

print("PASS: Temporary interaction loss tolerated.")


# -------------------------------------------------
# TEST 4D: Two separate repetitions
# -------------------------------------------------

rows = []

add_frames(rows, 0, 3, "GRASPING", 5.0)
add_frames(rows, 3, 7, "TRANSPORT", 10.0)
add_frames(rows, 10, 5, "NONE")

add_frames(rows, 15, 3, "GRASPING", 5.0)
add_frames(rows, 18, 7, "TRANSPORT", 10.0)
add_frames(rows, 25, 5, "NONE")

repetitions = segment_repetitions(rows)

assert len(repetitions) == 2
assert repetitions[0]["repetition_number"] == 1
assert repetitions[1]["repetition_number"] == 2

print("PASS: Two repetitions segmented correctly.")


# -------------------------------------------------
# TEST 4E: Insufficient movement
# -------------------------------------------------

rows = []

add_frames(rows, 0, 3, "GRASPING", 0.0)
add_frames(rows, 3, 7, "TRANSPORT", 1.0)
add_frames(rows, 10, 5, "NONE")

repetitions = segment_repetitions(rows)

assert len(repetitions) == 0

print("PASS: Insufficient movement rejected.")


# -------------------------------------------------
# TEST 4F: Recording ends during a repetition
# -------------------------------------------------

rows = []

add_frames(rows, 0, 3, "GRASPING", 5.0)
add_frames(rows, 3, 7, "TRANSPORT", 10.0)

repetitions = segment_repetitions(rows)

assert len(repetitions) == 1
assert repetitions[0]["ended_with_recording"] is True

print("PASS: Recording-end repetition identified.")


print("\nTEST 4 PASSED: All repetition segmentation checks succeeded.")


# Verify movement from all three confirmation frames.
rows = [
    make_row(0, "GRASPING", 5.0),
    make_row(1, "GRASPING", 5.0),
    make_row(2, "GRASPING", 5.0),
]

for frame in range(3, 10):
    rows.append(make_row(frame, "TRANSPORT", 10.0))

for frame in range(10, 15):
    rows.append(make_row(frame, "NONE"))

repetitions = segment_repetitions(rows)

assert len(repetitions) == 1
assert repetitions[0]["total_object_movement"] == 85.0

print("PASS: Movement accumulated across all grasp-confirmation frames.")




from src.assessment.repetition_assessment import (
    create_repetition_assessment,
)

# Confirm the recording-end flag survives assessment creation.
incomplete_repetition = {
    "repetition_number": 1,
    "start_frame": 0,
    "end_frame": 9,
    "start_timestamp_ms": 0,
    "end_timestamp_ms": 900,
    "duration_ms": 900,
    "total_object_movement": 85.0,
    "ended_with_recording": True,
}

assessment = create_repetition_assessment(
    repetition=incomplete_repetition
)

assert assessment["timing"]["ended_with_recording"] is True

print("PASS: Recording-end flag preserved in assessment.")





# Confirm different objects cannot share grasp-confirmation frames.
identity_rows = []

for frame in range(10):
    identity_rows.append({
        "frame_number": frame,
        "timestamp_ms": frame * 100,
        "interaction_state": (
            "GRASPING" if frame < 3 else "TRANSPORT"
        ),
        "object_class": "bottle",
        "object_id": 1 if frame < 2 else 2,
        "object_movement": 10.0,
    })

identity_repetitions = segment_repetitions(identity_rows)

assert identity_repetitions == [], (
    "Grasp confirmation incorrectly combined two object IDs."
)

print("PASS: Different objects cannot share grasp confirmation.")






from src.assessment.trajectory_assessment import (
    TrajectoryAssessment,
)
from src.assessment.repetition_processor import (
    process_repetition,
)

# Two objects observed within the same repetition window.
trajectory = TrajectoryAssessment(
    start_point=(0, 0),
    target_center=(100, 0),
    target_radius=10,
)

trajectory.trajectory_samples = [
    {"timestamp_ms": 0, "object_id": 1, "center": (0, 0)},
    {"timestamp_ms": 100, "object_id": 2, "center": (500, 500)},
    {"timestamp_ms": 200, "object_id": 1, "center": (50, 0)},
    {"timestamp_ms": 300, "object_id": 2, "center": (600, 600)},
    {"timestamp_ms": 400, "object_id": 1, "center": (100, 0)},
]

repetition = {
    "repetition_number": 1,
    "start_frame": 0,
    "end_frame": 4,
    "start_timestamp_ms": 0,
    "end_timestamp_ms": 400,
    "duration_ms": 400,
    "total_object_movement": 100.0,
    "object_id": 1,
}

assessment = process_repetition(
    repetition=repetition,
    pose_frames=None,
    trajectory_assessment=trajectory,
)

metrics = assessment["trajectory"]

assert metrics["sample_count"] == 3
assert metrics["path_length_px"] == 100.0
assert metrics["movement_efficiency"] == 1.0
assert metrics["target_reached"] is True

print("PASS: Trajectory assessment isolated to the correct object.")






# Missing object identity must not combine multiple object paths.
unknown_repetition = {
    **repetition,
    "object_id": None,
}

unknown_assessment = process_repetition(
    repetition=unknown_repetition,
    pose_frames=None,
    trajectory_assessment=trajectory,
)

assert unknown_assessment["trajectory"] is None

print("PASS: Ambiguous object identity does not produce trajectory metrics.")






waypoint_trajectory = TrajectoryAssessment(
    start_point=(0, 0),
    target_center=(100, 0),
    waypoints=[(25, 20), (75, 20)],
)


print(
    "DEBUG - Actual desired path:",
    waypoint_trajectory.desired_path,
)


assert waypoint_trajectory.desired_path == [
    (0, 0),
    (25, 20),
    (75, 20),
    (100, 0),
]

print("PASS: Desired trajectory contains no duplicate waypoints or target.")





reset_trajectory = TrajectoryAssessment(
    start_point=(0, 0),
    target_center=(100, 0),
)

reset_trajectory.start_attempt(object_id=1)
reset_trajectory.update(
    object_center=(20, 0),
    object_id=1,
    timestamp_ms=100,
)

assert len(reset_trajectory.trajectory_samples) == 1

reset_trajectory.end_attempt()
reset_trajectory.start_attempt(object_id=2)

assert reset_trajectory.actual_path == []
assert len(reset_trajectory.trajectory_samples) == 1
assert reset_trajectory.active_object_id == 2

reset_trajectory.reset()

assert reset_trajectory.actual_path == []
assert reset_trajectory.trajectory_samples == []
assert reset_trajectory.attempt_active is False
assert reset_trajectory.attempt_completed is False
assert reset_trajectory.active_object_id is None
assert reset_trajectory.inactive_frame_count == 0

print("PASS: Trajectory attempts retain history and session reset clears state.")