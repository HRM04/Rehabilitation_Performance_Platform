import csv
from pathlib import Path
from src.assessment.movement_quality import (
    calculate_range_of_motion,
    calculate_angular_velocity,
    calculate_angular_acceleration,
    count_velocity_reversals,
    calculate_trunk_center,
    calculate_trunk_displacement,
)
# Find the latest recorded pose CSV.
pose_files = list(
    Path("data/raw/pose").glob("*_pose_landmarks.csv")
)

if not pose_files:
    raise FileNotFoundError("No pose recordings found.")

pose_path = max(
    pose_files,
    key=lambda path: path.stat().st_mtime
)


# Read the recorded joint angles.
joint_names = [
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
]

joint_angles = {
    joint_name: []
    for joint_name in joint_names
}
frame_timestamps = []
# Store the landmarks for each recorded frame.
pose_frames = {}

with pose_path.open(newline="") as file:
    reader = csv.DictReader(file)

    # Each frame has multiple landmark rows.
    # Process each frame only once to avoid duplicate angle samples.
    processed_frames = set()

    for row in reader:
        landmark_name = row["landmark_name"]

        pose_frames.setdefault(row["frame_number"], {})[landmark_name] = {
            "x": float(row["x_norm"]),
            "y": float(row["y_norm"]),
            "visibility": float(row["visibility"]),
        }
    
        frame_number = row["frame_number"]

        if frame_number in processed_frames:
            continue

        processed_frames.add(frame_number)
        frame_timestamps.append(int(row["timestamp_ms"]))

        for joint_name in joint_names:
            value = row.get(f"{joint_name}_angle")

            angle = (
                float(value)
                if value not in ("", "None", None)
                else None
            )

            joint_angles[joint_name].append(angle)

# Store one summary record for each joint.
movement_results = []
# Calculate the observed angular range for each joint.
print(f"\nRecording: {pose_path.name}")

for joint_name in joint_names:
    result = calculate_range_of_motion(
        joint_angles[joint_name]
    )

    print(
        f"{joint_name}: "
        f"Min = {result['min_angle']}, "
        f"Max = {result['max_angle']}, "
        f"ROM = {result['range_of_motion']}"
    )

    # Calculate angular velocity between consecutive observations.
    angles = joint_angles[joint_name]
    velocities = []

        # Keep one velocity per frame interval, including missing values.
    # This preserves alignment with the recorded timestamps.
    interval_velocities = []

    for index in range(1, len(angles)):
        velocity = calculate_angular_velocity(
            previous_angle=angles[index - 1],
            current_angle=angles[index],
            previous_timestamp_ms=frame_timestamps[index - 1],
            current_timestamp_ms=frame_timestamps[index],
        )

        interval_velocities.append(velocity)

        if velocity is not None:
            velocities.append(velocity)

    if velocities:
        mean_speed = sum(abs(v) for v in velocities) / len(velocities)
        peak_speed = max(abs(v) for v in velocities)

        print(f"  Mean angular speed: {mean_speed:.2f} deg/s")
        print(f"  Peak angular speed: {peak_speed:.2f} deg/s")

    else:
        print("  Angular speed: insufficient valid measurements")


    # Calculate acceleration between consecutive velocity intervals.
    accelerations = []
    # Count movement-direction changes using the recorded velocities.
    reversal_count = count_velocity_reversals(
        interval_velocities,
        noise_threshold=5.0,
    )

    print(f"  Angular velocity reversals: {reversal_count}")

    for index in range(1, len(interval_velocities)):
        acceleration = calculate_angular_acceleration(
            previous_velocity=interval_velocities[index - 1],
            current_velocity=interval_velocities[index],
            previous_timestamp_ms=frame_timestamps[index],
            current_timestamp_ms=frame_timestamps[index + 1],
        )

        if acceleration is not None:
            accelerations.append(acceleration)

    if accelerations:
        mean_abs_acceleration = (
            sum(abs(a) for a in accelerations) / len(accelerations)
        )

        print(
            f"  Mean absolute angular acceleration: "
            f"{mean_abs_acceleration:.2f} deg/s²"
        )
    else:
        print("  Angular acceleration: insufficient valid measurements")


    movement_results.append({
    "recording": pose_path.name,
    "joint": joint_name,
    "min_angle_deg": result["min_angle"],
    "max_angle_deg": result["max_angle"],
    "observed_rom_deg": result["range_of_motion"],
    "mean_angular_speed_deg_s": (
        mean_speed if velocities else None
    ),
    "peak_angular_speed_deg_s": (
        peak_speed if velocities else None
    ),
    "mean_abs_angular_acceleration_deg_s2": (
        mean_abs_acceleration if accelerations else None
    ),
    "velocity_reversals": reversal_count,
})



# -----------------------------------------
# Save movement-assessment summary
# -----------------------------------------

output_dir = Path("data/processed/movement")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
f"{pose_path.stem}_movement_summary.csv"
)

if movement_results:
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(movement_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(movement_results)

print(f"\nMovement summary saved to: {output_path}")




# -----------------------------------------
# Analyze trunk displacement
# -----------------------------------------

baseline_center = None
trunk_displacements = []

for frame_number in sorted(pose_frames, key=int):
    trunk_center = calculate_trunk_center(
        pose_frames[frame_number]
    )

    if trunk_center is None:
        continue

    if baseline_center is None:
        baseline_center = trunk_center

    displacement = calculate_trunk_displacement(
        baseline_center,
        trunk_center,
    )

    if displacement is not None:
        trunk_displacements.append(displacement)

if trunk_displacements:
    print(
        f"Maximum observed trunk displacement: "
        f"{max(trunk_displacements):.4f} normalized image units"
    )
else:
    print("Trunk displacement: insufficient valid landmarks")



# -----------------------------------------
# Save trunk-displacement summary
# -----------------------------------------

trunk_summary_path = output_dir / (
    f"{pose_path.stem}_trunk_summary.csv"
)

with trunk_summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "recording",
            "maximum_trunk_displacement_norm",
            "valid_trunk_frames",
        ],
    )

    writer.writeheader()

    writer.writerow({
        "recording": pose_path.name,
        "maximum_trunk_displacement_norm": (
            max(trunk_displacements)
            if trunk_displacements
            else None
        ),
        "valid_trunk_frames": len(trunk_displacements),
    })

print(f"Trunk summary saved to: {trunk_summary_path}")