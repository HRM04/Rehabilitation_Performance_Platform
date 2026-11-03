from src.assessment.movement_quality import (
    calculate_range_of_motion,
    calculate_angular_velocity,
    calculate_angular_acceleration,
    count_velocity_reversals,
    calculate_trunk_center,
    calculate_trunk_displacement,
)


from src.assessment.posture_metrics import (
    calculate_shoulder_elevation,
    calculate_shoulder_vertical_displacement,
    calculate_trunk_posture,
)


from src.assessment.temporal_posture import (
    analyze_posture_deviations,
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
    reference_posture=None,
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
    # Use the validated pre-task calibration when supplied.
    # Otherwise, use the first valid frame of this repetition.
    reference_landmarks = (
        reference_posture.get("landmarks")
        if isinstance(reference_posture, dict)
        else None
    )
    baseline_center = (
        calculate_trunk_center(reference_landmarks)
        if reference_landmarks is not None
        else None
    )
    trunk_baseline_type = (
        "calibrated_reference"
        if baseline_center is not None
        else "first_valid_frame"
    )
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

    # Temporal trunk movement descriptors.
    # These quantify movement, not confirmed compensation.
    import statistics

    peak_index = (
        max(
            range(len(trunk_displacements)),
            key=trunk_displacements.__getitem__,
        )
        if trunk_displacements
        else None
    )

    result["trunk"] = {
        "baseline_type": trunk_baseline_type,
        "measurement_type": "relative_2d_trunk_center_displacement",
        "maximum_displacement_norm": (
            max(trunk_displacements)
            if trunk_displacements
            else None
        ),
        "valid_frames": len(trunk_displacements),
        "mean_displacement_norm": (
            statistics.mean(trunk_displacements)
            if trunk_displacements
            else None
        ),
        "displacement_std_norm": (
            statistics.pstdev(trunk_displacements)
            if trunk_displacements
            else None
        ),
        "peak_position_fraction": (
            peak_index / (len(trunk_displacements) - 1)
            if peak_index is not None
            and len(trunk_displacements) > 1
            else None
        ),
    }


    # -----------------------------------------
    # Shoulder-only vertical displacement
    # -----------------------------------------
    # Independent of hip landmarks.
    # Image-plane descriptor, not confirmed compensation.

    shoulder_vertical_baseline = (
        reference_posture.get("landmarks")
        if isinstance(reference_posture, dict)
        else None
    )

    shoulder_vertical_baseline_type = (
        "calibrated_reference"
        if shoulder_vertical_baseline is not None
        else "first_valid_frame"
    )

    shoulder_vertical_values = {
        "left": [],
        "right": [],
    }

    shoulder_vertical_samples = {
        "left": [],
        "right": [],
    }

    for frame in repetition_frames:
        landmarks = frame.get("landmarks")

        if shoulder_vertical_baseline is None:
            initial = calculate_shoulder_vertical_displacement(
                landmarks,
                landmarks,
            )
            if initial is not None:
                shoulder_vertical_baseline = landmarks

        displacement = (
            calculate_shoulder_vertical_displacement(
                shoulder_vertical_baseline,
                landmarks,
            )
            if shoulder_vertical_baseline is not None
            else None
        )

        for side in ("left", "right"):
            value = (
                displacement.get(
                    f"{side}_vertical_displacement_norm"
                )
                if displacement is not None
                else None
            )

            shoulder_vertical_samples[side].append({
                "timestamp_ms": frame["timestamp_ms"],
                "value": value,
            })

            if value is not None:
                shoulder_vertical_values[side].append(value)

    result["shoulder_vertical_displacement"] = {
        "measurement_type": (
            "absolute_image_plane_shoulder_vertical_displacement"
        ),
        "baseline_type": shoulder_vertical_baseline_type,
        "compensation_confirmed": None,
        "confirmation_status": "NOT_VALIDATED",
        "limitations": [
            "Shoulder movement may reflect trunk or whole-body movement",
            "Camera movement can affect image-plane displacement",
            "Not equivalent to shoulder elevation relative to hips",
        ],
    }

    for side in ("left", "right"):
        values = shoulder_vertical_values[side]
        upward = [max(0.0, v) for v in values]

        result["shoulder_vertical_displacement"][side] = {
            "valid_frames": len(values),
            "coverage_percent": (
                round(
                    100 * len(values) / len(repetition_frames),
                    1,
                )
                if repetition_frames
                else 0.0
            ),
            "maximum_upward_displacement_norm": (
                max(upward) if upward else None
            ),
            "mean_upward_displacement_norm": (
                sum(upward) / len(upward)
                if upward else None
            ),
            "maximum_absolute_displacement_norm": (
                max(map(abs, values))
                if values else None
            ),
            "temporal_analysis": analyze_posture_deviations(
                shoulder_vertical_samples[side],
                threshold=0.03,
            ),
        }

    # -----------------------------------------
    # Shoulder elevation relative to baseline
    # -----------------------------------------
    baseline_posture = (
        reference_posture.get("landmarks")
        if reference_posture is not None
        else None
    )
    shoulder_values = {
        "left": [],
        "right": [],
    }
    shoulder_samples = {
        "left": [],
        "right": [],
    }

    for frame in repetition_frames:
        landmarks = frame.get("landmarks")

        if baseline_posture is None:
            initial = calculate_shoulder_elevation(
                landmarks, landmarks
            )
            if initial is not None:
                baseline_posture = landmarks

        if baseline_posture is None:
            for side in ("left", "right"):
                shoulder_samples[side].append({
                    "timestamp_ms": frame["timestamp_ms"],
                    "value": None,
                })
            continue

        elevation = calculate_shoulder_elevation(
            baseline_posture, landmarks
        )

        if elevation is None:
            for side in ("left", "right"):
                shoulder_samples[side].append({
                    "timestamp_ms": frame["timestamp_ms"],
                    "value": None,
                })
            continue

        for side in ("left", "right"):
            value = elevation[
                f"{side}_shoulder_elevation_norm"
            ]

            shoulder_values[side].append(value)
            shoulder_samples[side].append({
                "timestamp_ms": frame["timestamp_ms"],
                "value": max(0.0, value),
            })

    result["shoulder_elevation"] = {
        "baseline_type": (
            "calibrated_reference"
            if reference_posture is not None
            else "first_valid_frame"
        ),
        "measurement_type": "relative_shoulder_hip_geometry",
        "valid_frames": len(shoulder_values["left"]),
    }

    for side in ("left", "right"):
        values = shoulder_values[side]
        upward_values = [max(0.0, value) for value in values]

        result["shoulder_elevation"][side] = {
            "maximum_upward_change_norm": (
                max(upward_values) if upward_values else None
            ),
            "mean_upward_change_norm": (
                sum(upward_values) / len(upward_values)
                if upward_values else None
            ),
            "temporal_analysis": analyze_posture_deviations(
                shoulder_samples[side],
                threshold=0.03,
            ),
        }

    # -----------------------------------------
    # Trunk posture measurements
    # -----------------------------------------
    baseline_posture = (
        reference_posture.get("landmarks")
        if reference_posture is not None
        else None
    )
    lateral_shifts = []
    inclination_changes = []
    inclination_samples = []
    lateral_samples = []

    for frame in repetition_frames:
        landmarks = frame.get("landmarks")

        if baseline_posture is None:
            initial = calculate_trunk_posture(
                landmarks, landmarks
            )
            if initial is not None:
                baseline_posture = landmarks

        if baseline_posture is None:
            for target in (inclination_samples, lateral_samples):
                target.append({
                    "timestamp_ms": frame["timestamp_ms"],
                    "value": None,
                })
            continue

        posture = calculate_trunk_posture(
            baseline_posture, landmarks
        )

        if posture is None:
            for target in (inclination_samples, lateral_samples):
                target.append({
                    "timestamp_ms": frame["timestamp_ms"],
                    "value": None,
                })
            continue

        lateral_shifts.append(
            posture["lateral_shift_norm"]
        )
        inclination_changes.append(
            posture["inclination_change_deg"]
        )

        inclination_samples.append({
            "timestamp_ms": frame["timestamp_ms"],
            "value": posture["inclination_change_deg"],
        })
        lateral_samples.append({
            "timestamp_ms": frame["timestamp_ms"],
            "value": posture["lateral_shift_norm"],
        })

    result["trunk_posture"] = {
        "measurement_type": "relative_2d_trunk_posture",
        "baseline_type": (
            "calibrated_reference"
            if reference_posture is not None
            else "first_valid_frame"
        ),
        "valid_frames": len(lateral_shifts),

        "maximum_lateral_shift_norm": (
            max(map(abs, lateral_shifts))
            if lateral_shifts else None
        ),

        "maximum_inclination_change_deg": (
            max(map(abs, inclination_changes))
            if inclination_changes else None
        ),

        "mean_inclination_change_deg": (
            sum(map(abs, inclination_changes))
            / len(inclination_changes)
            if inclination_changes else None
        ),

        "lateral_shift_temporal_analysis": (
            analyze_posture_deviations(
                lateral_samples,
                threshold=0.03,
            )
        ),
        "signed_lateral_shifts_norm": lateral_shifts,
        "signed_inclination_changes_deg": inclination_changes,

        "inclination_temporal_analysis": (
            analyze_posture_deviations(
                inclination_samples,
                threshold=10.0,
            )
        ),
    }

    return result