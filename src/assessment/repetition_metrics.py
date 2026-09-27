from src.assessment.movement_quality import (
    calculate_range_of_motion,
    calculate_angular_velocity,
    calculate_angular_acceleration,
    count_velocity_reversals,
    calculate_trunk_center,
    calculate_trunk_displacement,
)


JOINT_NAMES = [
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
]


def calculate_repetition_metrics(
    pose_frames,
    start_timestamp_ms,
    end_timestamp_ms,
):
    """
    Calculate movement metrics for one repetition window.

    pose_frames should contain one entry per frame with:
        timestamp_ms
        landmarks
        angles
    """

    repetition_frames = [
        frame
        for frame in pose_frames
        if start_timestamp_ms
        <= frame["timestamp_ms"]
        <= end_timestamp_ms
    ]

    if len(repetition_frames) < 2:
        return None

    result = {
        "duration_ms": (
            end_timestamp_ms - start_timestamp_ms
        ),
        "joints": {},
        "trunk": {},
    }

    # -----------------------------------------
    # Joint-level metrics
    # -----------------------------------------
    for joint_name in JOINT_NAMES:

        angles = [
            frame["angles"].get(joint_name)
            for frame in repetition_frames
        ]

        timestamps = [
            frame["timestamp_ms"]
            for frame in repetition_frames
        ]

        rom = calculate_range_of_motion(angles)

        interval_velocities = []
        valid_velocities = []

        for index in range(1, len(angles)):

            velocity = calculate_angular_velocity(
                previous_angle=angles[index - 1],
                current_angle=angles[index],
                previous_timestamp_ms=timestamps[index - 1],
                current_timestamp_ms=timestamps[index],
            )

            interval_velocities.append(velocity)

            if velocity is not None:
                valid_velocities.append(velocity)

        accelerations = []

        for index in range(1, len(interval_velocities)):

            acceleration = calculate_angular_acceleration(
                previous_velocity=interval_velocities[index - 1],
                current_velocity=interval_velocities[index],
                previous_timestamp_ms=timestamps[index],
                current_timestamp_ms=timestamps[index + 1],
            )

            if acceleration is not None:
                accelerations.append(acceleration)

        result["joints"][joint_name] = {
            "min_angle_deg": rom["min_angle"],
            "max_angle_deg": rom["max_angle"],
            "rom_deg": rom["range_of_motion"],

            "mean_angular_speed_deg_s": (
                sum(abs(v) for v in valid_velocities)
                / len(valid_velocities)
                if valid_velocities
                else None
            ),

            "peak_angular_speed_deg_s": (
                max(abs(v) for v in valid_velocities)
                if valid_velocities
                else None
            ),

            "mean_abs_angular_acceleration_deg_s2": (
                sum(abs(a) for a in accelerations)
                / len(accelerations)
                if accelerations
                else None
            ),

            "velocity_reversals": count_velocity_reversals(
                interval_velocities,
                noise_threshold=5.0,
            ),
        }

    # -----------------------------------------
    # Trunk displacement
    # -----------------------------------------
    baseline_center = None
    trunk_displacements = []

    for frame in repetition_frames:

        trunk_center = calculate_trunk_center(
            frame["landmarks"]
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

    result["trunk"] = {
        "maximum_displacement_norm": (
            max(trunk_displacements)
            if trunk_displacements
            else None
        ),
        "valid_frames": len(trunk_displacements),
    }

    return result