
from src.assessment.trajectory_assessment import TrajectoryAssessment

# Create a simulated rehabilitation task.
trajectory = TrajectoryAssessment(
    start_point=(250, 400),
    target_center=(1000, 400),
    target_radius=70,
)

# Simulated bottle positions along a straight path.
positions = [
    (250, 400),
    (350, 400),
    (450, 400),
    (550, 400),
    (650, 400),
    (750, 400),
    (850, 400),
    (950, 400),
    (1000, 400),
]

# Begin the simulated attempt.
trajectory.start_attempt(object_id=1)

for frame_number, position in enumerate(positions):
    trajectory.update(
        object_center=position,
        object_id=1,
        timestamp_ms=frame_number * 100,
    )

# Simulate the interaction ending.
for _ in range(10):
    trajectory.update_attempt_state(
        interaction_active=False,
        grace_frames=10,
    )

metrics = trajectory.get_metrics()

print("\nOFFLINE TRAJECTORY TEST")
print("-----------------------")

for name, value in metrics.items():
    print(f"{name}: {value}")

# Check expected results.
assert trajectory.attempt_active is False
assert len(trajectory.actual_path) == len(positions)
assert abs(metrics["path_length_px"] - 750) < 0.001
assert abs(metrics["direct_displacement_px"] - 750) < 0.001
assert abs(metrics["movement_efficiency"] - 1.0) < 0.001
assert abs(metrics["mean_trajectory_deviation_px"]) < 0.001
assert abs(metrics["final_target_error_px"]) < 0.001
assert metrics["target_reached"] is True

print("\nPASS: All trajectory checks succeeded.")







# -------------------------------------------------
# TEST 3: Detection loss and object identity
# -------------------------------------------------

print("\nTEST 3: DETECTION LOSS AND OBJECT IDENTITY")
print("------------------------------------------")

trajectory = TrajectoryAssessment(
    start_point=(250, 400),
    target_center=(1000, 400),
    target_radius=70,
)

trajectory.start_attempt(object_id=1)

# Record the initial movement of object 1.
for i, position in enumerate([
    (250, 400),
    (350, 400),
    (450, 400),
]):
    trajectory.update(
        object_center=position,
        object_id=1,
        timestamp_ms=i * 100,
    )

assert len(trajectory.actual_path) == 3

# Simulate five frames of missing interaction evidence.
for _ in range(5):
    ended = trajectory.update_attempt_state(
        interaction_active=False,
        grace_frames=10,
    )
    assert ended is False

assert trajectory.attempt_active is True
print("PASS: Short detection loss did not end the attempt.")

# Same object reappears.
trajectory.update_attempt_state(
    interaction_active=True,
    grace_frames=10,
)

trajectory.update(
    object_center=(550, 400),
    object_id=1,
    timestamp_ms=800,
)

assert len(trajectory.actual_path) == 4
assert trajectory.inactive_frame_count == 0

print("PASS: Tracking resumed for the original object.")

# A different object is detected.
trajectory.update(
    object_center=(900, 600),
    object_id=2,
    timestamp_ms=900,
)

assert len(trajectory.actual_path) == 4
assert trajectory.actual_path[-1] == (550, 400)

print("PASS: Different object ID was ignored.")

# Simulate prolonged inactivity.
for _ in range(10):
    ended = trajectory.update_attempt_state(
        interaction_active=False,
        grace_frames=10,
    )

assert ended is True
assert trajectory.attempt_active is False

print("PASS: Prolonged inactivity ended the attempt.")

# Verify recorded trajectory remains intact.
assert len(trajectory.actual_path) == 4

metrics = trajectory.get_metrics()

assert abs(metrics["path_length_px"] - 300) < 0.001
assert metrics["target_reached"] is False

print("PASS: Recorded trajectory was preserved.")
print("\nTEST 3 PASSED: All checks succeeded.")
