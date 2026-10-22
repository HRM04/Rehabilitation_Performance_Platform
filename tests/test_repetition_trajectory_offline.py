
from src.assessment.trajectory_assessment import TrajectoryAssessment
from src.assessment.repetition_processor import process_repetition


trajectory = TrajectoryAssessment(
    start_point=(250, 400),
    target_center=(1000, 400),
    target_radius=70,
)

# -------------------------------------------------
# Repetition 1: Straight path, successful arrival
# -------------------------------------------------

trajectory.start_attempt(object_id=1)

first_path = [
    (250, 400),
    (400, 400),
    (550, 400),
    (700, 400),
    (850, 400),
    (1000, 400),
]

for i, position in enumerate(first_path):
    trajectory.update(
        object_center=position,
        object_id=1,
        timestamp_ms=i * 100,
    )

trajectory.end_attempt()


# -------------------------------------------------
# Repetition 2: Deviated path, incomplete arrival
# -------------------------------------------------

trajectory.start_attempt(object_id=1)

second_path = [
    (250, 400),
    (350, 500),
    (450, 550),
    (550, 500),
    (650, 450),
    (750, 400),
]

for i, position in enumerate(second_path):
    trajectory.update(
        object_center=position,
        object_id=1,
        timestamp_ms=1000 + i * 100,
    )

trajectory.end_attempt()


# -------------------------------------------------
# Create corresponding repetition records
# -------------------------------------------------

repetition_1 = {
    "repetition_number": 1,
    "start_frame": 0,
    "end_frame": 5,
    "start_timestamp_ms": 0,
    "end_timestamp_ms": 500,
    "duration_ms": 500,
    "total_object_movement": 750.0,
}

repetition_2 = {
    "repetition_number": 2,
    "start_frame": 10,
    "end_frame": 15,
    "start_timestamp_ms": 1000,
    "end_timestamp_ms": 1500,
    "duration_ms": 500,
    "total_object_movement": 600.0,
}


# -------------------------------------------------
# Process repetitions independently
# -------------------------------------------------

assessment_1 = process_repetition(
    repetition=repetition_1,
    pose_frames=None,
    trajectory_assessment=trajectory,
)

assessment_2 = process_repetition(
    repetition=repetition_2,
    pose_frames=None,
    trajectory_assessment=trajectory,
)

metrics_1 = assessment_1["trajectory"]
metrics_2 = assessment_2["trajectory"]

print("\nTEST 5: REPETITION–TRAJECTORY INTEGRATION")
print("-----------------------------------------")

print("\nRepetition 1:")
for name, value in metrics_1.items():
    print(f"{name}: {value}")

print("\nRepetition 2:")
for name, value in metrics_2.items():
    print(f"{name}: {value}")


# -------------------------------------------------
# Assertions
# -------------------------------------------------

assert assessment_1["repetition_number"] == 1
assert assessment_2["repetition_number"] == 2

assert metrics_1["target_reached"] is True
assert metrics_2["target_reached"] is False

assert abs(metrics_1["movement_efficiency"] - 1.0) < 0.001
assert metrics_2["movement_efficiency"] < 1.0

assert abs(metrics_1["final_target_error_px"]) < 0.001
assert metrics_2["final_target_error_px"] > 70

assert metrics_1["mean_trajectory_deviation_px"] == 0.0
assert metrics_2["mean_trajectory_deviation_px"] > 0.0

assert assessment_1["movement_quality"] is None
assert assessment_2["movement_quality"] is None

print("\nPASS: Repetitions have independent trajectory metrics.")
print("PASS: Successful and incomplete movements distinguished.")
print("PASS: Missing pose data handled correctly.")
print("\nTEST 5 PASSED: All integration checks succeeded.")
