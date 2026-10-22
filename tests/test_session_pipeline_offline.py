
from src.assessment.session_pipeline import process_session
from src.assessment.trajectory_assessment import TrajectoryAssessment


def make_row(frame, state, movement=0.0):
    return {
        "frame_number": frame,
        "timestamp_ms": frame * 100,
        "interaction_state": state,
        "object_class": "bottle",
        "object_movement": movement,
    }


# -------------------------------------------------
# 1. Simulate two rehabilitation repetitions
# -------------------------------------------------

rows = []

for start_frame in (0, 20):
    for offset in range(3):
        rows.append(make_row(start_frame + offset, "GRASPING", 5.0))

    for offset in range(3, 10):
        rows.append(make_row(start_frame + offset, "TRANSPORT", 100.0))

    for offset in range(10, 15):
        rows.append(make_row(start_frame + offset, "NONE"))


# -------------------------------------------------
# 2. Simulate trajectory observations
# -------------------------------------------------

trajectory = TrajectoryAssessment(
    start_point=(250, 400),
    target_center=(1000, 400),
    target_radius=70,
)

paths = [
    [
        (250, 400),
        (400, 400),
        (550, 400),
        (700, 400),
        (850, 400),
        (1000, 400),
    ],
    [
        (250, 400),
        (350, 500),
        (450, 550),
        (550, 500),
        (650, 450),
        (750, 400),
    ],
]

for start_frame, path in zip((0, 20), paths):
    trajectory.start_attempt(object_id=1)

    for i, position in enumerate(path):
        trajectory.update(
            object_center=position,
            object_id=1,
            timestamp_ms=(start_frame + i) * 100,
        )

    trajectory.end_attempt()


# -------------------------------------------------
# 3. Run the full session pipeline
# -------------------------------------------------

result = process_session(
    session_id="offline_test_session_001",
    task_id="bottle_transport",
    interaction_rows=rows,
    pose_frames=None,
    trajectory_assessment=trajectory,
    patient_id="test_patient",
)

session = result["session_assessment"]
state = result["rehab_state"]

repetitions = session["repetitions"]
assessments = session["repetition_assessments"]

print("\nTEST 6: COMPLETE OFFLINE SESSION PIPELINE")
print("-----------------------------------------")
print("Detected repetitions:", len(repetitions))
print("Generated assessments:", len(assessments))

for assessment in assessments:
    print(
        "Repetition",
        assessment["repetition_number"],
        "| Target reached:",
        assessment["trajectory"]["target_reached"],
        "| Efficiency:",
        assessment["trajectory"]["movement_efficiency"],
    )


# -------------------------------------------------
# 4. Verify pipeline results
# -------------------------------------------------

assert len(repetitions) == 2
assert len(assessments) == 2

assert assessments[0]["trajectory"]["target_reached"] is True
assert assessments[1]["trajectory"]["target_reached"] is False

assert assessments[0]["trajectory"]["movement_efficiency"] == 1.0
assert assessments[1]["trajectory"]["movement_efficiency"] < 1.0

assert session["movement_consistency"] is None
assert session["performance_deterioration"] is None

assert isinstance(state, dict)
assert result["candidate_baseline"] is not None

print("\nPASS: Two repetitions detected.")
print("PASS: Individual assessments generated.")
print("PASS: Trajectory results correctly associated.")
print("PASS: Missing pose measurements handled correctly.")
print("PASS: Rehabilitation state created.")
print("PASS: Candidate baseline generated.")

print("\nTEST 6 PASSED: Complete offline pipeline succeeded.")
