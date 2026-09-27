import math


def calculate_2d_distance(point_a, point_b):
    """
    Calculate Euclidean distance between two 2D points.

    Points should be dictionaries containing x and y.
    """

    if point_a is None or point_b is None:
        return None

    dx = point_b["x"] - point_a["x"]
    dy = point_b["y"] - point_a["y"]

    return math.hypot(dx, dy)


def calculate_path_length(object_positions):
    """
    Calculate the total observed object path length.

    Missing positions break continuity so that movement is not
    estimated across unobserved gaps.
    """

    if not object_positions:
        return None

    total_distance = 0.0
    valid_segments = 0

    previous_position = None

    for position in object_positions:

        if position is None:
            previous_position = None
            continue

        if previous_position is not None:

            distance = calculate_2d_distance(
                previous_position,
                position,
            )

            if distance is not None:
                total_distance += distance
                valid_segments += 1

        previous_position = position

    if valid_segments == 0:
        return None

    return total_distance


def calculate_movement_efficiency(object_positions):
    """
    Calculate preliminary object-transport efficiency.

    Efficiency =
        straight-line displacement / observed path length

    Values closer to 1 indicate a more direct path.

    This is an engineering movement metric, not a validated
    clinical efficiency score.
    """

    valid_positions = [
        position
        for position in object_positions
        if position is not None
    ]

    if len(valid_positions) < 2:
        return None

    path_length = calculate_path_length(
        object_positions
    )

    if path_length is None or path_length == 0:
        return None

    direct_displacement = calculate_2d_distance(
        valid_positions[0],
        valid_positions[-1],
    )

    if direct_displacement is None:
        return None

    efficiency = direct_displacement / path_length

    # Numerical noise should not produce values slightly > 1.
    efficiency = min(1.0, efficiency)

    return {
        "direct_displacement": direct_displacement,
        "path_length": path_length,
        "movement_efficiency": efficiency,
    }


def assess_transport_execution(
    object_positions,
):
    """
    Summarize the observed object transport trajectory.
    """

    efficiency_metrics = calculate_movement_efficiency(
        object_positions
    )

    if efficiency_metrics is None:
        return {
            "trajectory_available": False,
            "direct_displacement": None,
            "path_length": None,
            "movement_efficiency": None,
        }

    return {
        "trajectory_available": True,
        **efficiency_metrics,
    }